"""Build and screen grip-coherent references on the shared rigid-bar model."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.configuration import structured_config
from bello_mujoco.evaluation.barbell import BarbellConfig, solve_pose, foot_support
from bello_mujoco.evaluation.statics import static_equilibrium
from bello_mujoco.models import repo_root
from bello_mujoco.models.barbell import (
    GRIP_ANCHORS,
    grip_axis_errors,
    grip_wrench_columns,
)


def smoothstep(value):
    return value**3 * (10 + value * (-15 + 6 * value))


def trajectory(
    robot, excursion, pitch=0.0, *, seconds=4.0, fps=50.0, constrain_wrist=False
):
    """Hold->offset->hold, using Cartesian IK on a quintic time curve."""
    config = structured_config(
        BarbellConfig, OmegaConf.load(repo_root() / "configs/analysis/barbell.yaml")
    )
    model = robot.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = robot.default_qpos
    mujoco.mj_forward(model, data)
    qpos = []
    count = round(seconds * fps)
    phases = np.r_[
        np.zeros(round(fps)),
        smoothstep(np.linspace(0, 1, count + 1)),
        np.ones(round(fps)),
        smoothstep(np.linspace(1, 0, count + 1)),
        np.zeros(round(fps)),
    ]
    bar = int(model.joint("barbell_free").qposadr[0])
    for phase in phases:
        pose = replace(
            config,
            bar_center_in_torso_m=tuple(
                np.array(config.bar_center_in_torso_m) + phase * np.asarray(excursion)
            ),
            hand_rpy_rad=config.hand_rpy_rad,
        )
        wrist_targets = None
        if constrain_wrist:
            wrist_targets = {
                side: robot.default_qpos[
                    int(model.joint(f"{side}_wrist_pitch_joint").qposadr[0])
                ]
                + phase * pitch
                for side in ("left", "right")
            }
        center, _ = solve_pose(
            robot, data, pose, report_visual=False, wrist_pitch_targets=wrist_targets
        )
        torso = model.body(config.torso_body).id
        data.qpos[bar : bar + 3] = center
        data.qpos[bar + 3 : bar + 7] = data.xquat[torso]
        qpos.append(data.qpos.copy())
    qpos = np.asarray(qpos)
    qvel = np.zeros((len(qpos), model.nv))
    for i in range(1, len(qpos) - 1):
        mujoco.mj_differentiatePos(model, qvel[i], 2 / fps, qpos[i - 1], qpos[i + 1])
    qacc = np.gradient(qvel, 1 / fps, axis=0)
    return qpos, qvel, qacc


def screen(robot, qpos, qvel, qacc, *, stride=1, loaded=True):
    """Inverse-dynamics feasibility with vertical sole reactions and shaft-axis hinges.

    Solves full floating-body equations with original motor and joint effort
    limits. This is a necessary feasibility screen, not a PD tracking claim.
    """
    model = robot.mj_model
    data = mujoco.MjData(model)
    if not loaded:
        data.eq_active[robot.grip_equality_ids] = False
    motor_map = np.zeros((model.nv, model.nu))
    motor_map[robot.actuator_qvel_indices] = robot.actuator_coordinate_transform.T
    motor_bounds = model.actuator_forcerange
    joint_ids = robot.actuator_joint_ids
    limited = model.jnt_actfrclimited[joint_ids].astype(bool)
    effort_map = np.r_[np.eye(model.nu), robot.actuator_coordinate_transform.T[limited]]
    effort_bounds = np.r_[motor_bounds, model.jnt_actfrcrange[joint_ids[limited]]]
    worst = 0.0
    gap = 0.0
    axis_error = 0.0
    penetration = 0.0
    torques = []
    failures = []
    foot_ids = set(robot.foot_body_ids.tolist())
    mass_matrix = np.zeros((model.nv, model.nv))
    for frame in range(0, len(qpos), stride):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        points, _ = foot_support(robot, data)
        support_columns = []
        mu = float(np.min(model.geom_friction[robot.foot_geom_ids, 0]))
        for point in points:
            body = int(
                robot.foot_body_ids[
                    np.argmin(
                        np.linalg.norm(
                            data.xpos[robot.foot_body_ids, :2] - point[:2], axis=1
                        )
                    )
                ]
            )
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, body)
            for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                support_columns.append(
                    jac.T @ [x * mu / np.sqrt(2), y * mu / np.sqrt(2), 1]
                )
        support = np.column_stack(support_columns)
        grip_columns = []
        if loaded:
            axis_error = max(axis_error, float(np.max(grip_axis_errors(model, data))))
        for side in ("left", "right") if loaded else ():
            for anchor in GRIP_ANCHORS:
                hand, bar = (
                    model.site(f"{side}_{kind}_{anchor}").id for kind in ("hand", "bar")
                )
                gap = max(
                    gap, np.linalg.norm(data.site_xpos[hand] - data.site_xpos[bar])
                )
            reactions = grip_wrench_columns(model, data, side)
            grip_columns.extend([reactions, -reactions])
        support = np.concatenate([support, *grip_columns], axis=1)
        mujoco.mj_fullM(model, data, mass_matrix)
        required = mass_matrix @ qacc[frame] + data.qfrc_bias - data.qfrc_passive
        rows = np.arange(model.nv)
        if not loaded:
            bar_v = int(model.joint("barbell_free").dofadr[0])
            rows = rows[(rows < bar_v) | (rows >= bar_v + 6)]
        solution = static_equilibrium(
            motor_map[rows], support[rows], required[rows], effort_map, effort_bounds
        )
        if solution is None:
            failures.append(frame)
        else:
            worst = max(worst, solution.maximum_limit_fraction)
            torques.append(solution.motor_torque)
        for c in data.contact:
            bodies = model.geom_bodyid[[c.geom1, c.geom2]]
            if not loaded and model.body("barbell").id in bodies:
                continue
            if 0 in bodies and any(b in foot_ids for b in bodies):
                continue
            penetration = max(penetration, -float(c.dist))
    joint = qpos[:, robot.actuator_qpos_indices]
    limits = robot.target_limits.joint
    violation = max(
        0.0, float(np.max(limits[:, 0] - joint)), float(np.max(joint - limits[:, 1]))
    )
    return dict(
        passed=not failures
        and worst <= 1.0
        and axis_error <= 0.01
        and gap <= 0.002
        and penetration <= 0.001
        and violation <= 1e-6,
        maximum_effort_fraction=float(worst),
        max_grip_error_m=float(gap),
        max_grip_axis_error_rad=axis_error,
        max_unintended_penetration_m=penetration,
        max_joint_violation_rad=violation,
        no_equilibrium_frames=failures,
        peak_motor_torque_nm=np.max(np.abs(torques), axis=0).tolist()
        if torques
        else [],
        mass_kg=float(model.body("barbell").mass[0]),
    )


def make_clip(robot, qpos, qvel, fps, passed):
    from bello_mujoco.motion.clip import MotionClip

    model = robot.mj_model
    data = mujoco.MjData(model)
    from bello_mujoco.robot_tasks import motion_task_definition

    names = (
        *motion_task_definition(robot.robot.asset).beyondmimic_tracked_body_names,
        "barbell",
    )
    body_ids = [model.body(name).id for name in names]
    positions, quaternions, linear, angular = [], [], [], []
    for q, v in zip(qpos, qvel, strict=True):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_ids].copy())
        quaternions.append(data.xquat[body_ids].copy())
        velocities = []
        for body in body_ids:
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
            )
            velocities.append(velocity)
        velocities = np.asarray(velocities)
        angular.append(velocities[:, :3])
        linear.append(velocities[:, 3:])
    return MotionClip(
        fps=fps,
        joint_names=robot.actuator_joint_names,
        body_names=names,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=np.asarray(positions, dtype=np.float32),
        body_quat=np.asarray(quaternions, dtype=np.float32),
        body_lin_vel=np.asarray(linear, dtype=np.float32),
        body_ang_vel=np.asarray(angular, dtype=np.float32),
        foot_contacts=np.ones((len(qpos), 2), dtype=np.float32),
        object_mass_kg=float(model.body_mass[robot.bar_body_id]),
        screen_passed=passed,
        free_joint_qpos=tuple(q for q, _ in robot.free_joint_coordinates),
        joint_qpos_indices=tuple(robot.actuator_qpos_indices.tolist()),
        joint_qvel_indices=tuple(robot.actuator_qvel_indices.tolist()),
    )


def render_reference(robot, qpos, fps, output):
    import imageio_ffmpeg

    model = robot.mj_model
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.0, 0.0, 0.7]
    camera.distance = 2.5
    camera.azimuth = 135.0
    camera.elevation = -12.0
    writer = imageio_ffmpeg.write_frames(str(output), (960, 720), fps=fps)
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            for frame in qpos:
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.send(renderer.render())
    finally:
        writer.close()


def main():
    import argparse
    import json
    from bello_mujoco.configuration import (
        locomotion_training_profile,
        motion_training_profile,
    )
    from bello_mujoco.sim import load_model
    from bello_mujoco.terrain import FLAT_TERRAIN
    from bello_mujoco.motion.clip import save_motion_clip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/barbell-training"))
    parser.add_argument(
        "--standing", action="store_true", help="Also screen standing pose trajectories"
    )
    parser.add_argument(
        "--video", action="store_true", help="Render the motion candidate for review"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = motion_training_profile(name="bello_mini_barbell_motion")
    robot = load_model(
        config.asset_config,
        terrain=FLAT_TERRAIN,
        spawn_clearance=0,
        use_viewer_model=args.video,
        load_mjx=False,
    )
    report = dict(
        motor_names=robot.actuator_coordinate_names,
        motor_peak_limits_nm=robot.actuator_peak_force.tolist(),
        robot_mass_kg=float(
            robot.mj_model.body_mass.sum() - robot.mj_model.body_mass[robot.bar_body_id]
        ),
        bar_mass_kg=float(robot.mj_model.body_mass[robot.bar_body_id]),
        candidates=[],
    )
    # Sample at the policy rate; endpoint holds and each transition remain smooth.
    fps = 50.0
    selected = None
    for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
        excursion = np.array([-0.03, 0.0, 0.08]) * scale
        try:
            qpos, qvel, qacc = trajectory(robot, excursion, fps=fps)
        except ValueError as error:
            report["candidates"].append(
                dict(excursion_m=excursion.tolist(), passed=False, error=str(error))
            )
            continue
        result = screen(robot, qpos, qvel, qacc)
        result["excursion_m"] = excursion.tolist()
        if scale == 0:
            result["passed"] = False
            result["reason"] = "zero excursion is diagnostic only"
        report["candidates"].append(result)
        print(json.dumps(result), flush=True)
        if selected is None:
            selected = qpos, qvel, result
        if result["passed"]:
            selected = qpos, qvel, result
            break
        # A failed endpoint hold cannot be fixed by shortening the excursion.
        hold = screen(robot, qpos[:1], np.zeros_like(qvel[:1]), np.zeros_like(qacc[:1]))
        if not hold["passed"]:
            report["endpoint_hold"] = hold
            break
    if selected is None:
        report["training_ready"] = False
        (args.output / "diagnostics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        raise ValueError(
            "no grip-coherent candidate could be generated; see diagnostics.json"
        )
    qpos, qvel, result = selected
    save_motion_clip(
        args.output / "lift.npz", make_clip(robot, qpos, qvel, fps, result["passed"])
    )
    report["training_ready"] = bool(result["passed"])
    report["approval_required"] = True
    if args.video:
        render_reference(robot, qpos, fps, args.output / "lift.mp4")
    if args.standing:
        standing_config = locomotion_training_profile("bello_mini_loaded_standing")
        standing = load_model(
            standing_config.asset_config,
            terrain=FLAT_TERRAIN,
            spawn_clearance=0,
            load_mjx=False,
        )
        hold_qpos = np.asarray(standing.default_qpos)[None]
        hold_qvel = np.zeros((1, standing.mj_model.nv))
        report["standing_endpoint_hold"] = screen(
            standing, hold_qpos, hold_qvel, hold_qvel
        )
    if args.standing and not report["standing_endpoint_hold"]["passed"]:
        report["standing_training_ready"] = False
        report["standing_blocker"] = (
            "The shared 30 kg endpoint hold failed; pose-bank generation is blocked."
        )
    elif args.standing:
        variants = [
            (x, z, pitch)
            for x in (-0.02, 0.0, 0.02)
            for z in (-0.03, 0.0, 0.03)
            for pitch in (-np.pi / 18, 0.0, np.pi / 18)
        ]
        targets, loaded, screens = [], [], []
        bar_body = standing.mj_model.body("barbell").id
        full_inertia = standing.mj_model.body_inertia[bar_body].copy()
        for x, z, pitch in variants:
            try:
                q, v, a = trajectory(
                    standing, [x, 0.0, z], pitch, fps=fps, constrain_wrist=True
                )
            except ValueError as error:
                screens.append(
                    dict(
                        offset_m=[x, 0.0, z],
                        wrist_pitch_rad=pitch,
                        passed=False,
                        error=str(error),
                    )
                )
                continue
            load_results = []
            for mass in (5.0, 10.0, 20.0, 30.0):
                standing.mj_model.body_mass[bar_body] = mass
                standing.mj_model.body_inertia[bar_body] = full_inertia * mass / 30.0
                mujoco.mj_setConst(standing.mj_model, mujoco.MjData(standing.mj_model))
                load_results.append(screen(standing, q, v, a))
            targets.append(
                q[:, standing.actuator_qpos_indices]
                @ standing.actuator_coordinate_transform.T
            )
            loaded.append(True)
            screens.append(
                dict(
                    offset_m=[x, 0.0, z],
                    wrist_pitch_rad=pitch,
                    passed=all(r["passed"] for r in load_results),
                    loads=load_results,
                )
            )
        if not targets:
            report["standing"] = screens
            report["standing_training_ready"] = False
            (args.output / "diagnostics.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            raise ValueError(
                "no coherent standing grip trajectories; see diagnostics.json"
            )
        # Unloaded rest excursions use the same smooth timing and original limits.
        from bello_mujoco.actuators import ActuatorConfig

        rest_config = ActuatorConfig.for_asset("bello_mini")
        rest = np.array(
            [
                rest_config.joints[name].default_pos
                for name in standing.actuator_joint_names
            ]
        )
        base = standing.default_qpos.copy()
        count = targets[0].shape[0]
        phase = np.r_[
            np.zeros(round(fps)),
            smoothstep(np.linspace(0, 1, round(4 * fps) + 1)),
            np.ones(round(fps)),
            smoothstep(np.linspace(1, 0, round(4 * fps) + 1)),
            np.zeros(round(fps)),
        ]
        for fraction in (0.5, 1.0):
            q = np.repeat(base[None], count, axis=0)
            held = standing.hold_ctrl_indices
            indices = standing.actuator_qpos_indices[held]
            q[:, indices] += phase[:, None] * fraction * (rest[held] - base[indices])
            # Rest trajectories have no active bar; park it and screen robot dynamics.
            bar = standing.bar_qpos_adr
            q[:, bar : bar + 3] = [10.0, 0.0, 0.15]
            targets.append(
                q[:, standing.actuator_qpos_indices]
                @ standing.actuator_coordinate_transform.T
            )
            loaded.append(False)
            velocities = np.zeros((count, standing.mj_model.nv))
            for frame in range(1, count - 1):
                mujoco.mj_differentiatePos(
                    standing.mj_model,
                    velocities[frame],
                    2 / fps,
                    q[frame - 1],
                    q[frame + 1],
                )
            screens.append(
                dict(
                    rest_fraction=fraction,
                    **screen(
                        standing,
                        q,
                        velocities,
                        np.gradient(velocities, 1 / fps, axis=0),
                        loaded=False,
                    ),
                )
            )
        passed = all(s["passed"] for s in screens)
        np.savez_compressed(
            args.output / "standing.npz",
            targets=np.asarray(targets, dtype=np.float32),
            loaded=loaded,
            fps=fps,
            passed=passed,
            joint_names=np.asarray(standing.actuator_joint_names),
        )
        report["standing"] = screens
        report["standing_training_ready"] = passed
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
