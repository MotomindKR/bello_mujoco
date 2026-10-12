"""Historical-policy MuJoCo worker. Launched by bello-eval-barbell, not directly.

Imports resolve against the explicitly selected motion checkpoint checkout.
Standing uses its separately validated source and exported native actor.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from standing import NativeController
from bello_mujoco.motion.runtime import load_motion_policy_runtime
from bello_mujoco.models.urdf import ensure_imu_sensors, ImuSensorConfig
from bello_mujoco.evaluation.motion import native_motion_state, _native_motion_outcome
from bello_mujoco.policy import (
    apply_actor,
    init_observation_stack,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.training.normalization import normalize_observation
from bello_mujoco.envs.actions import limit_motor_step, project_motor_target
from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.envs.barbell import bar_metrics


class Simulation:
    def __init__(self, config):
        self.config = config
        self.out = Path(config["output_path"])
        self.md = json.loads(Path(config["standing_metadata"]).read_text())
        self.rt = load_motion_policy_runtime(
            Path(config["motion_checkpoint"]), use_viewer_model=True
        )
        self.env = env = self.rt.env
        md = self.md
        original = env.model.mj_model
        if list(env.model.actuator_coordinate_names) != md["coordinates"]:
            raise ValueError("Standing and motion actuator coordinates differ")
        if not np.allclose(
            env.model.actuator_coordinate_transform, md["motor_from_joint"]
        ):
            raise ValueError("Standing and motion coordinate transforms differ")
        self.dt = env.control_timestep
        if not np.isclose(self.dt, md["control_timestep"]):
            raise ValueError("Policies require different control timesteps")
        if not np.isclose(config["mode"]["motion_seconds"], env.motion.duration):
            raise ValueError(
                "Configured motion duration must match the complete checkpoint clip"
            )
        targets = json.loads(Path(config["neutral_pose_path"]).read_text())
        if targets["coordinates"] != md["coordinates"]:
            raise ValueError("Arm target coordinates do not match policies")
        self.normal = np.asarray(targets["motor_positions"])
        self.forward = np.asarray(md["default_position"])
        self.held = np.setdiff1d(np.arange(original.nu), md["policy_indices"])
        mujoco.mj_saveLastXML(str(self.out / "base.xml"), original)
        xml = ET.parse(self.out / "base.xml").getroot()
        ensure_imu_sensors(
            xml,
            ImuSensorConfig(enabled=True, body_name=md["hardware_imu_frame"]),
            env.model.robot,
        )
        ET.SubElement(
            xml.find("worldbody"),
            "body",
            name="external_bar_holder",
            mocap="true",
            pos="0 0 0",
        )
        ET.SubElement(
            xml.find("equality"),
            "weld",
            name="external_handover_support",
            body1="barbell",
            body2="external_bar_holder",
            relpose="0 0 0 1 0 0 0",
            solref="0.01 1",
            solimp="0.99 0.999 0.001",
            active="false",
        )
        for key in xml.findall("keyframe"):
            xml.remove(key)
        ET.ElementTree(xml).write(
            self.out / "scene.xml", encoding="utf-8", xml_declaration=True
        )
        self.m = m = mujoco.MjModel.from_xml_path(str(self.out / "scene.xml"))
        if (m.nq, m.nv, m.nu) != (original.nq, original.nv, original.nu):
            raise ValueError("Holder changed robot dimensions")
        object.__setattr__(env, "model", replace(env.model, mj_model=m))
        self.d = d = mujoco.MjData(m)
        d.qpos[:] = env.model.default_qpos
        d.qpos[env.model.actuator_qpos_indices] = (
            np.asarray(md["joint_from_motor"]) @ self.normal
        )
        d.ctrl[:] = self.normal
        self.barq = env.model.bar_qpos_adr
        self.barbody = env.model.bar_body_id
        self.grips = env.model.grip_equality_ids
        self.support = m.equality("external_handover_support").id
        d.qpos[self.barq : self.barq + 7] = env.motion.qpos[
            0, self.barq : self.barq + 7
        ]
        d.qpos[self.barq] += 0.45
        d.mocap_pos[0] = d.qpos[self.barq : self.barq + 3]
        d.mocap_quat[0] = d.qpos[self.barq + 3 : self.barq + 7]
        d.eq_active[self.grips] = False
        d.eq_active[self.support] = True
        self.standing_gain = np.asarray(md["kp"])
        self.standing_damping = np.asarray(md["kd"])
        self.motion_gain = original.actuator_gainprm[:, 0].copy()
        self.motion_damping = -original.actuator_biasprm[:, 2].copy()
        self.set_gains(0.0)
        mujoco.mj_forward(m, d)
        mujoco.mj_saveModel(m, str(self.out / "scene.mjb"))
        probe = mujoco.MjData(m)
        probe.qpos[:] = env.model.default_qpos
        probe.eq_active[:] = False
        for name, value in md["environment"]["arm_reset_joint_positions"]:
            probe.qpos[int(m.joint(name).qposadr[0])] = value
        mujoco.mj_forward(m, probe)
        motor_pose = (
            np.asarray(md["motor_from_joint"])
            @ probe.qpos[env.model.actuator_qpos_indices]
        )
        gravity = np.linalg.solve(
            np.asarray(md["motor_from_joint"]).T,
            probe.qfrc_bias[env.model.actuator_qvel_indices],
        )
        self.unloaded = self.forward.copy()
        self.unloaded[self.held] = (
            motor_pose[self.held] + gravity[self.held] / self.standing_gain[self.held]
        )
        self.weight_force = -float(m.body_mass[self.barbody]) * m.opt.gravity
        if not np.isclose(m.body_mass[self.barbody], config["bar_mass_kg"]):
            raise ValueError("Requested bar mass does not match checkpoint scene")
        stand_limits = TargetLimits(
            np.asarray(md["motor_limits"]),
            np.asarray(md["joint_limits"]),
            np.asarray(md["joint_from_motor"]),
            np.asarray(md["position_step_limits"]),
        )
        for pose in (self.normal, self.forward, self.unloaded):
            stand_limits.validate_reference(pose)
        limits = env.model.target_limits
        envelope = TargetLimits(
            np.stack(
                [
                    np.minimum(limits.motor[:, 0], stand_limits.motor[:, 0]),
                    np.maximum(limits.motor[:, 1], stand_limits.motor[:, 1]),
                ],
                axis=1,
            ),
            np.stack(
                [
                    np.minimum(limits.joint[:, 0], stand_limits.joint[:, 0]),
                    np.maximum(limits.joint[:, 1], stand_limits.joint[:, 1]),
                ],
                axis=1,
            ),
            limits.joint_from_motor,
            np.minimum(limits.position_step, stand_limits.position_step),
        )
        self.project = jax.jit(
            lambda target, prev: limit_motor_step(
                project_motor_target(target, jnp.asarray(self.forward), envelope),
                prev,
                envelope,
            )
        )
        self.target = jax.jit(lambda state, action: env.action_target(state, action))
        self.outcome = jax.jit(
            lambda old, new, key: _native_motion_outcome(env, old, new, key)
        )
        self.measure = jax.jit(lambda data: bar_metrics(env.model, data))
        self.actor = jax.jit(
            lambda obs: apply_actor(
                self.rt.policy,
                normalize_observation(
                    obs, self.rt.normalization.observation, self.rt.metadata.ppo
                ),
            )
        )
        self.log = (self.out / "standing.log").open("w")
        self.controller = self.make_standing(self.normal)
        self.previous = d.ctrl.copy()
        self.rng = jax.random.PRNGKey(0)
        self.state = None
        self.stack = None
        self.phase = None
        self.rows = []
        self.qtrace, self.vtrace, self.ctrace, self.etrace = [], [], [], []

    def make_standing(self, reference):
        settings = SimpleNamespace(
            policy=Path(self.config["standing_policy"]),
            source=Path(self.config["standing_source"]),
            binary=Path(self.config["native_binary"]),
            observation_delay_steps=0,
        )
        return NativeController(settings, self.md, self.m, reference, self.log)

    def set_gains(self, weight):
        kp = (1 - weight) * self.standing_gain + weight * self.motion_gain
        kd = (1 - weight) * self.standing_damping + weight * self.motion_damping
        self.m.actuator_gainprm[:, 0] = kp
        self.m.actuator_biasprm[:, 1] = -kp
        self.m.actuator_biasprm[:, 2] = -kd

    def begin_phase(self, phase):
        d, m = self.d, self.m
        self.phase = phase
        if phase == "await_bar":
            self.holder_initial = d.mocap_pos[0].copy()
            points = np.array(
                [
                    d.site_xpos[m.site(f"{side}_hand_grip").id]
                    for side in ("left", "right")
                ]
            )
            self.holder_goal = points.mean(axis=0)
            vector = points[0] - points[1]
            local = (
                m.site_pos[m.site("left_bar_grip").id]
                - m.site_pos[m.site("right_bar_grip").id]
            )
            quat = Rotation.align_vectors(
                [vector / np.linalg.norm(vector)], [local / np.linalg.norm(local)]
            )[0].as_quat()
            d.mocap_quat[0] = quat[[3, 0, 1, 2]]
        if phase == "transfer_load":
            d.eq_active[self.grips] = True
            d.eq_active[self.support] = False
            d.xfrc_applied[self.barbody, :3] = self.weight_force
            mujoco.mj_forward(m, d)
        if phase == "to_motion":
            yaw = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")[2]
            self.state = native_motion_state(
                self.env,
                d,
                last_action=np.zeros(self.env.action_size),
                reference_time=jnp.asarray(0.0),
                alignment_xy=jnp.asarray(d.qpos[:2]),
                alignment_yaw=jnp.asarray(yaw),
                joint_default_offset=jnp.zeros(self.env.action_size),
                step_count=0,
                episode_return=0.0,
            )
            self.rng, feedback_rng, delay_rng, obs_rng = jax.random.split(self.rng, 4)
            feedback = self.env._noisy_actor_feedback(self.state.data, feedback_rng)
            self.state = self.state.replace(
                actor_feedback_history=jnp.repeat(
                    feedback[None, :],
                    self.env.maximum_observation_delay_steps + 1,
                    axis=0,
                ),
                actor_feedback_delay_steps=self.env._sample_observation_delay(
                    delay_rng
                ),
            )
            self.stack = init_observation_stack(
                self.env.observation(self.state, obs_rng),
                self.rt.metadata.ppo.policy.obs_stack_size,
            )
        if phase == "to_standing":
            self.controller.close()
            self.controller = self.make_standing(self.forward)
            self.controller.previous_target = self.previous.copy()
        if phase == "open_hands":
            # Full gravity support was established before opening; maintain it
            # until the simulated fingers have reached open and the bar is clear.
            d.mocap_pos[0] = d.qpos[self.barq : self.barq + 3]
            d.mocap_quat[0] = d.qpos[self.barq + 3 : self.barq + 7]
            d.eq_active[self.support] = True
            d.eq_active[self.grips] = False
            d.xfrc_applied[self.barbody] = 0
            mujoco.mj_forward(m, d)
        if phase == "await_bar_clear":
            self.holder_initial = d.mocap_pos[0].copy()
            self.holder_goal = self.holder_initial + np.array([0.5, 0, 0.05])

    def step(self, command):
        env, m, d = self.env, self.m, self.d
        phase = command["phase"]
        if phase != self.phase:
            self.begin_phase(phase)
        load, weight = command["load_fraction"], command["motion_weight"]
        d.xfrc_applied[self.barbody] = 0
        if phase in ("transfer_load", "transfer_to_holder", "interrupted") and np.any(
            d.eq_active[self.grips]
        ):
            d.xfrc_applied[self.barbody, :3] = (1 - load) * self.weight_force
        arm_reference = self.unloaded + (self.forward - self.unloaded) * load
        self.controller.reference = (
            self.normal + (arm_reference - self.normal) * command["arm_fraction"]
        )
        use_motion = (
            phase in ("to_motion", "motion", "to_standing") and self.state is not None
        )
        action = np.zeros(env.action_size)
        if use_motion:
            action = self.actor(stacked_observation(self.stack))
            motion_target = np.asarray(self.target(self.state, action))
        if phase == "motion":
            requested = motion_target
        else:
            standing_target = self.controller(d)
            requested = (
                (1 - weight) * standing_target + weight * motion_target
                if use_motion
                else standing_target
            )
        requested = np.asarray(
            self.project(jnp.asarray(requested), jnp.asarray(self.previous))
        )
        self.set_gains(weight)
        if phase in ("await_bar", "await_bar_clear"):
            fraction = min(
                1.0, command["elapsed_seconds"] / self.config["holder_move_seconds"]
            )
            smooth = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
            d.mocap_pos[0] = (
                self.holder_initial + (self.holder_goal - self.holder_initial) * smooth
            )
        delay = (
            env.config.command_delay_seconds
            if phase in ("to_motion", "motion")
            else self.md["environment"]["command_delay_seconds"]
        )
        for sub in range(env.decimation):
            fraction = np.clip(
                ((sub + 1) * m.opt.timestep - delay) / m.opt.timestep, 0, 1
            )
            d.ctrl[:] = self.previous + (requested - self.previous) * fraction
            mujoco.mj_step(m, d)
        self.previous = requested.copy()
        stepped = native_motion_state(
            env,
            d,
            last_action=np.asarray(action),
            reference_time=jnp.asarray(command["motion_time"]),
            alignment_xy=self.state.alignment_xy
            if self.state is not None
            else jnp.zeros(2),
            alignment_yaw=self.state.alignment_yaw
            if self.state is not None
            else jnp.asarray(0.0),
            joint_default_offset=jnp.zeros(env.action_size),
            step_count=len(self.rows) + 1,
            episode_return=0.0,
        )
        info = {}
        if use_motion:
            self.rng, key = jax.random.split(self.rng)
            self.state, obs, _, _, info = self.outcome(self.state, stepped, key)
            self.stack = update_observation_stack(self.stack, obs, jnp.asarray(False))
        metrics = {
            key: float(value) for key, value in self.measure(stepped.data).items()
        }
        up = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).apply([0, 0, 1])
        tilt = float(np.arccos(np.clip(up[2], -1, 1)))
        forearm = (
            d.xanchor[m.joint("left_wrist_yaw_joint").id]
            - d.xanchor[m.joint("left_elbow_pitch_joint").id]
        )
        forearm_angle = float(
            np.rad2deg(np.arccos(np.clip(-forearm[2] / np.linalg.norm(forearm), -1, 1)))
        )
        gaps = [
            float(
                np.linalg.norm(
                    d.site_xpos[m.site(f"{side}_hand_grip").id]
                    - d.site_xpos[m.site(f"{side}_bar_grip").id]
                )
            )
            for side in ("left", "right")
        ]
        row = dict(
            command,
            time_s=float(d.time),
            root_height_m=float(d.qpos[2]),
            root_tilt_rad=tilt,
            grips_active=bool(np.any(d.eq_active[self.grips])),
            external_support=bool(d.eq_active[self.support])
            or bool(np.linalg.norm(d.xfrc_applied[self.barbody, :3]) > 1e-6),
            external_support_force_n=d.xfrc_applied[self.barbody, :3].tolist(),
            forearm_angle_from_down_deg=forearm_angle,
            hand_bar_gap_m=max(gaps),
            torque_peak_ratio=float(
                np.max(np.abs(d.actuator_force) / env.model.actuator_peak_force)
            ),
            **metrics,
        )
        self.rows.append(row)
        self.qtrace.append(d.qpos.copy())
        self.vtrace.append(d.qvel.copy())
        self.ctrace.append(d.ctrl.copy())
        self.etrace.append(d.eq_active.copy())
        failure = None
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            failure = "nonfinite"
        elif phase != "motion" and (d.qpos[2] < 0.55 or tilt > 0.9):
            failure = "standing_fall"
        elif metrics["barbell/unintended_contact_force_n"] > 1:
            failure = "unintended_bar_contact"
        elif phase == "motion" and any(
            bool(info.get(key, False))
            for key in (
                "done/height",
                "done/orientation",
                "done/end_effector",
                "done/nonfinite",
            )
        ):
            failure = "motion_tracking_terminal"
        row["failure"] = failure
        return row

    def close(self):
        try:
            self.controller.close()
        finally:
            self.log.close()
            np.savez_compressed(
                self.out / "cycle_trace.npz",
                qpos=np.asarray(self.qtrace),
                qvel=np.asarray(self.vtrace),
                ctrl=np.asarray(self.ctrace),
                eq_active=np.asarray(self.etrace),
                dt=self.dt,
            )
            (self.out / "cycle_trace.json").write_text(json.dumps(self.rows, indent=2))


def main():
    output = sys.stdout
    sys.stdout = sys.stderr  # Library/compiler diagnostics cannot corrupt JSON IPC.
    config = json.loads(Path(sys.argv[1]).read_text())
    simulation = Simulation(config)
    print(json.dumps({"ready": True, "dt": simulation.dt}), file=output, flush=True)
    try:
        for line in sys.stdin:
            result = simulation.step(json.loads(line))
            print(json.dumps(result, allow_nan=False), file=output, flush=True)
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
