"""Standing observation adapter for the explicitly checked historical workers."""

from collections import deque
import json
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from bello_mujoco.configuration import structured_config
from bello_mujoco.envs.actions import limit_motor_step, project_motor_target
from bello_mujoco.envs.feedback import ObservationLowPassFilterConfig
from bello_mujoco.envs.locomotion_config import ObservationConfig
from bello_mujoco.envs.observations import (
    filter_feedback,
    unpack_feedback,
    build_observation,
)
from bello_mujoco.imu import MujocoImu
from bello_mujoco.models.target_limits import TargetLimits


class NativeController:
    """Assemble standing observations; run the exported actor with native ONNX."""

    def __init__(self, config, metadata, model, initial_target, log):
        self.metadata = metadata
        self.last_action = np.zeros(len(metadata["policy_indices"]), dtype=np.float32)
        self.observation_history = deque(maxlen=metadata["observation_stack_size"])
        self.obs_config = structured_config(
            ObservationConfig, OmegaConf.create(metadata["environment"]["observation"])
        )
        assert metadata["command"]["lin_vel_x"] == [0.0, 0.0]
        assert metadata["command"]["lin_vel_y"] == [0.0, 0.0]
        assert metadata["command"]["ang_vel_yaw"] == [0.0, 0.0]
        self.joint_from_motor = np.asarray(metadata["joint_from_motor"])
        self.imu = MujocoImu.bind(model)
        self.previous_feedback = None
        self.history = deque(maxlen=config.observation_delay_steps + 1)
        self.held = np.setdiff1d(np.arange(model.nu), metadata["policy_indices"])
        self.reference = np.asarray(initial_target, dtype=np.float32)
        self.previous_target = self.reference.copy()
        self.filtered = structured_config(
            ObservationLowPassFilterConfig,
            OmegaConf.create(metadata["environment"]["observation_low_pass_filter"]),
        )
        self.filter = jax.jit(
            lambda old, new: filter_feedback(
                old, new, metadata["control_timestep"], self.filtered
            )
        )
        limits = TargetLimits(
            np.asarray(metadata["motor_limits"]),
            np.asarray(metadata["joint_limits"]),
            np.asarray(metadata["joint_from_motor"]),
            np.asarray(metadata["position_step_limits"]),
        )
        limits.validate_reference(initial_target)
        self.limit = jax.jit(
            lambda target, previous, reference: limit_motor_step(
                project_motor_target(target, reference, limits),
                previous,
                limits,
            )
        )
        self.constrained_commands = 0
        self.calls = 0
        self.log = log
        self.process = subprocess.Popen(
            [
                str(config.binary),
                "--policy",
                str(config.policy),
                "--source",
                str(config.source),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )

    def __call__(self, data):
        sample = self.imu.sample(data)
        rotation = Rotation.from_quat(sample.orientation_xyzw)
        gravity = rotation.inv().apply([0, 0, -1])
        measured = jnp.asarray(
            np.r_[
                sample.linear_acceleration_xyz,
                sample.angular_velocity_xyz,
                gravity,
                data.actuator_length,
                data.actuator_velocity,
            ],
            dtype=jnp.float32,
        )
        self.previous_feedback = (
            measured
            if self.previous_feedback is None
            else self.filter(self.previous_feedback, measured)
        )
        acc, gyro, filtered_gravity, pos, vel = map(
            np.asarray, unpack_feedback(self.previous_feedback)
        )
        # Match the native zero-command observation layout; the exported graph
        # includes the checkpoint's observation normalization and actor weights.
        joint_position = self.joint_from_motor @ pos - np.asarray(
            self.metadata["default_joint_position"]
        )
        joint_velocity = self.joint_from_motor @ vel
        arm = (self.joint_from_motor @ self.reference)[self.held]
        terms = tuple(
            jnp.asarray(x, dtype=jnp.float32)
            for x in (
                np.zeros(3),
                acc,
                gyro,
                filtered_gravity,
                np.zeros(1),
                np.zeros(3),
                joint_position,
                joint_velocity,
                np.zeros(len(pos)),
                self.last_action,
            )
        )
        obs = np.asarray(
            build_observation(
                terms,
                jnp.zeros(3),
                self.obs_config,
                None,
                torso_command=jnp.zeros(3),
                arm_target=jnp.asarray(arm),
            )
        )
        self.last_observation = obs.copy()
        if not self.observation_history:
            self.observation_history.extend([obs] * self.observation_history.maxlen)
        else:
            self.observation_history.append(obs)
        self.process.stdin.write(
            json.dumps(np.concatenate(self.observation_history).tolist()) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native policy stopped; inspect {self.log.name}")
        result = json.loads(line)
        self.last_action = np.clip(
            np.asarray(result["action"], dtype=np.float32),
            -self.metadata["action_clip"],
            self.metadata["action_clip"],
        )
        target = np.asarray(result["motor_target"], dtype=np.float32)
        target[self.held] = self.reference[self.held]
        limited = np.asarray(
            self.limit(
                jnp.asarray(target),
                jnp.asarray(self.previous_target),
                jnp.asarray(self.reference),
            )
        )
        self.constrained_commands += int(np.max(np.abs(limited - target)) > 1e-5)
        self.previous_target = limited
        self.calls += 1
        return limited

    def close(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError(
                "native diagnostic process did not exit after EOF"
            ) from None
        finally:
            self.process.stdout.close()
        if code:
            raise RuntimeError(
                f"native policy exited with {code}; inspect {self.log.name}"
            )
