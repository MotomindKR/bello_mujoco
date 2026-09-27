"""Opt-in SONIC observation/history parity against the compiled Rust runner."""

import json
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.deployment import sonic_actor_onnx, sonic_deployment_bundle
from bello_mujoco.motion.clip import MotionClip, save_motion_clip
from bello_mujoco.sonic.embodiment import (
    G1_JOINT_NAMES,
    MiniEmbodimentAdapter,
    g1_neutral_positions,
)
from bello_mujoco.sonic.model import (
    SonicActor,
    SonicActorCritic,
    SonicCritic,
    SonicMLP,
    SonicNetworkConfig,
)
from bello_mujoco.sonic.observations import SonicObservationAssembler
from bello_mujoco.sonic.reference import ReferenceBank

pytestmark = pytest.mark.integration


def test_rust_sonic_matches_jax_across_history_reset_and_clip_boundary(
    rust_policy, tmp_path
) -> None:
    run, _, source = rust_policy
    source_names = tuple(name for name in G1_JOINT_NAMES if name != "waist_pitch_joint")
    joint_names = tuple(
        name.replace("_elbow_joint", "_elbow_pitch_joint") for name in source_names
    )
    source_indices = np.asarray(
        [G1_JOINT_NAMES.index(name) for name in source_names], dtype=np.int32
    )
    source_to_target = np.full(len(G1_JOINT_NAMES), -1, dtype=np.int32)
    source_to_target[source_indices] = np.arange(len(joint_names))
    default = np.linspace(-0.15, 0.15, len(joint_names), dtype=np.float32)
    adapter = MiniEmbodimentAdapter(
        target_joint_names=joint_names,
        target_neutral=default,
        target_to_source_indices=source_indices,
        source_to_target_indices=source_to_target,
        signs=np.ones(len(joint_names), dtype=np.float32),
        source_neutral=g1_neutral_positions(),
        motor_from_joint=np.eye(len(joint_names), dtype=np.float32),
    )
    generator = np.random.default_rng(91)

    def mlp(input_size, output_size, rank, key):
        tensors = (
            (
                generator.normal(size=(output_size, input_size)).astype(np.float32)
                * 0.005,
                generator.normal(size=output_size).astype(np.float32) * 0.005,
            ),
        )
        return SonicMLP(tensors, rank=rank, alpha=float(rank), key=key)

    keys = jax.random.split(jax.random.PRNGKey(92), 3)
    network = SonicNetworkConfig()
    policy = SonicActorCritic(
        SonicActor(
            mlp(640, 64, 0, keys[0]),
            mlp(994, 29, 2, keys[1]),
            adapter,
            network,
        ),
        SonicCritic(mlp(1645, 1, 2, keys[2]), adapter, body_count=14),
        jnp.full(29, np.log(0.05), dtype=jnp.float32),
    )
    frames = 20
    qpos = np.zeros((frames, 7 + len(joint_names)), dtype=np.float32)
    qpos[:, 2] = 0.8
    root_yaw = np.arange(frames, dtype=np.float32) * 0.03
    qpos[:, 3] = np.cos(root_yaw / 2.0)
    qpos[:, 6] = np.sin(root_yaw / 2.0)
    qpos[:, 7:] = default + np.arange(frames)[:, None] * 0.001
    qvel = np.zeros((frames, 6 + len(joint_names)), dtype=np.float32)
    qvel[:, 6:] = np.arange(frames)[:, None] * 0.002
    body_quat = np.zeros((frames, 14, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    clip = MotionClip(
        fps=50.0,
        joint_names=joint_names,
        body_names=tuple(f"body_{index}" for index in range(14)),
        qpos=jnp.asarray(qpos),
        qvel=jnp.asarray(qvel),
        body_pos=jnp.zeros((frames, 14, 3)),
        body_quat=jnp.asarray(body_quat),
        body_lin_vel=jnp.zeros((frames, 14, 3)),
        body_ang_vel=jnp.zeros((frames, 14, 3)),
        foot_contacts=jnp.zeros((frames, 2)),
    )
    clip_path = tmp_path / "walk.npz"
    save_motion_clip(clip_path, clip)
    joint = SimpleNamespace(effort_limit=4.0, gains=(2.0, 0.1))
    limits = np.tile([-2.0, 2.0], (len(joint_names), 1)).astype(np.float32)
    model = SimpleNamespace(
        actuator_joint_names=joint_names,
        default_joint_target=default,
        actuator_coordinate_transform=np.eye(len(joint_names), dtype=np.float32),
        mj_model=SimpleNamespace(actuator_ctrlrange=limits),
    )
    env = SimpleNamespace(
        action_clip=100.0,
        actuator_config=SimpleNamespace(by_name={name: joint for name in joint_names}),
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assembler = SonicObservationAssembler(adapter)
    metadata = {
        "git_commit": commit,
        "task": "sonic_transfer",
        "robot": "synthetic",
        "observation_size": policy.actor.observation_size,
        "observation_frame": [
            ["sonic_future_reference", 640],
            ["gravity_direction", 3],
            ["base_angular_velocity", 3],
            ["joint_position", 28],
            ["joint_velocity", 28],
            ["last_action", 28],
        ],
        "observation_stack_size": 10,
        "observation_max_abs": 10000.0,
        "coordinates": list(joint_names),
        "policy_indices": list(range(len(joint_names))),
        "default_position": default.tolist(),
        "kp": [2.0] * len(joint_names),
        "kd": [0.1] * len(joint_names),
        "control_timestep": 0.02,
        "joint_from_motor": np.eye(len(joint_names)).tolist(),
        "default_joint_position": default.tolist(),
        "joint_names": list(joint_names),
        "joint_limits": limits.tolist(),
        "position_step_limits": [10.0] * len(joint_names),
        "motor_from_joint": np.eye(len(joint_names)).tolist(),
        "motor_limits": limits.tolist(),
        "environment": {},
        "action_clip": 100.0,
        "hardware_imu_frame": "imu",
        "root_from_hardware_imu_xyzw": [0.0, 0.0, 0.0, 1.0],
        "reference_clips": ["walk"],
        "future_frames": 10,
        "future_dt": 0.1,
        "source_joint_indices": source_indices.tolist(),
        "source_neutral": adapter.source_neutral.tolist(),
        "source_signs": adapter.signs.tolist(),
        "pretrained_revision": "a" * 40,
        "pretrained_sha256": "b" * 64,
        "dataset_artifact": "entity/project/dataset:v0",
    }
    graph = sonic_actor_onnx(policy, model=model, env=env, metadata=metadata)
    bundle = sonic_deployment_bundle(
        graph.SerializeToString(),
        physics=b"<mujoco/>",
        clips=(("walk", clip_path),),
    )
    samples = [
        {
            "motor_position": default.tolist(),
            "motor_velocity": [0.0] * len(joint_names),
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "angular_velocity_local": [0.1, -0.2, 0.3],
            "reference_time": reference_time,
            "initialize": initialize,
        }
        for reference_time, initialize in ((0.0, True), (0.017, False), (0.379, True))
    ]
    result = run(samples, content=bundle, telemetry=True, clip="walk")
    assert result.returncode == 0, result.stderr
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    bank = ReferenceBank.from_clips((("walk", clip, "train"),))
    history = None
    previous_action = np.zeros(len(joint_names), dtype=np.float32)
    alignment_yaw = 0.0
    for sample, output in zip(samples, outputs, strict=True):
        if sample["initialize"]:
            history = None
            previous_action.fill(0.0)
            initial = bank.sample_time(jnp.asarray(sample["reference_time"]))
            quaternion = np.asarray(initial.qpos[3:7])
            alignment_yaw = -float(
                np.arctan2(
                    2.0
                    * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                    1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
                )
            )
        local_frame = (sample["reference_time"] + 0.02) * clip.fps
        reference = assembler.reference(
            bank,
            jnp.asarray(0),
            jnp.asarray(local_frame),
            jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            jnp.asarray(alignment_yaw),
        )
        frame = np.asarray(
            assembler.step(
                reference,
                root_quaternion=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
                root_angular_velocity=jnp.asarray(sample["angular_velocity_local"]),
                logical_joint_position=jnp.zeros(len(joint_names)),
                logical_joint_velocity=jnp.zeros(len(joint_names)),
                previous_action=jnp.asarray(previous_action),
            )
        )
        if history is None:
            history = np.tile(frame, 10)
        else:
            history = np.concatenate((history[len(frame) :], frame))
        expected_action = np.asarray(policy.actor(jnp.asarray(history)))
        expected_applied = np.clip(expected_action, -100.0, 100.0)
        expected_motor = np.clip(default + 0.5 * expected_applied, -2.0, 2.0)
        np.testing.assert_allclose(output["observation"], history, atol=2.0e-6)
        np.testing.assert_allclose(output["raw_action"], expected_action, atol=2.0e-5)
        np.testing.assert_allclose(output["motor_target"], expected_motor, atol=2.0e-5)
        previous_action = expected_applied
