"""Export an ONNX actor and its reference data as one deployment bundle."""

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from bello_mujoco.policy import ActorCritic
from bello_mujoco.training.ppo_types import ObservationNormState
from bello_mujoco.training.ppo_config import PPOConfig


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: Path | None = None
    output_path: Path | None = None
    motion_dataset: Path | None = None
    reference_clips: tuple[str, ...] = ()


def actor_onnx(
    policy: ActorCritic,
    normalization: ObservationNormState,
    config: PPOConfig,
    *,
    default_ctrl,
    ctrl_limits,
    policy_indices,
    action_clip: float,
    metadata: dict,
    effort_scale=None,
):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, constants = [], []

    def constant(name, value):
        constants.append(
            numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)
        )
        return name

    def op(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], **attributes))
        return output

    observation_size = int(policy.actor.layers[0].weight.shape[1])
    value = "observation"
    if config.normalize_observations:
        variance = np.asarray(normalization.var)
        denominator = (
            np.sqrt(np.maximum(variance, 1e-8)) + config.observation_norm_eps
            if config.observation_norm_epsilon_mode == "std_offset"
            else np.sqrt(np.maximum(variance, config.observation_norm_eps))
        )
        value = op(
            "Sub", [value, constant("observation_mean", normalization.mean)], "centered"
        )
        value = op(
            "Div",
            [value, constant("observation_denominator", denominator)],
            "normalized",
        )
        value = op(
            "Clip",
            [
                value,
                constant("normalization_low", -config.observation_norm_clip),
                constant("normalization_high", config.observation_norm_clip),
            ],
            "normalized_clipped",
        )
    activations = {"tanh": "Tanh", "relu": "Relu", "elu": "Elu"}
    for index, layer in enumerate(policy.actor.layers):
        value = op(
            "Gemm",
            [
                value,
                constant(f"weight_{index}", np.asarray(layer.weight).T),
                constant(f"bias_{index}", layer.bias),
            ],
            f"linear_{index}",
        )
        if index != len(policy.actor.layers) - 1:
            if config.policy.activation == "swish":
                sigmoid = op("Sigmoid", [value], f"sigmoid_{index}")
                value = op("Mul", [value, sigmoid], f"activation_{index}")
            else:
                value = op(
                    activations[config.policy.activation],
                    [value],
                    f"activation_{index}",
                )
    if policy.action_distribution == "tanh_gaussian":
        value = op("Tanh", [value], "squashed_action")
        value = op(
            "Mul",
            [value, constant("distribution_limit", policy.action_limit)],
            "action",
        )
    else:
        value = op("Identity", [value], "action")

    default = np.asarray(default_ctrl, dtype=np.float32)
    limits = np.asarray(ctrl_limits, dtype=np.float32)
    indices = np.asarray(policy_indices)
    scatter = np.eye(len(default), dtype=np.float32)[indices]
    if effort_scale is None:
        default = np.clip(default, limits[:, 0], limits[:, 1])
        scaled = op(
            "Div", [value, constant("action_clip", action_clip)], "normalized_action"
        )
        bounded = op(
            "Clip",
            [scaled, constant("negative_one", -1.0), constant("one", 1.0)],
            "bounded_action",
        )
        positive = op(
            "GreaterOrEqual", [bounded, constant("zero", 0.0)], "positive_action"
        )
        span = op(
            "Where",
            [
                positive,
                constant("positive_span", limits[indices, 1] - default[indices]),
                constant("negative_span", default[indices] - limits[indices, 0]),
            ],
            "target_span",
        )
        delta = op("Mul", [bounded, span], "motor_delta")
    else:
        value = op(
            "Clip",
            [
                value,
                constant("effort_action_low", -action_clip),
                constant("effort_action_high", action_clip),
            ],
            "applied_effort_action",
        )
        delta = op(
            "Mul", [value, constant("effort_scale", effort_scale)], "motor_delta"
        )
    all_delta = op(
        "MatMul", [delta, constant("policy_coordinates", scatter)], "all_motor_delta"
    )
    target = op(
        "Add", [all_delta, constant("default_ctrl", default)], "requested_motor_target"
    )
    target = op(
        "Max", [target, constant("motor_low", limits[:, 0])], "motor_lower_bounded"
    )
    op("Min", [target, constant("motor_high", limits[:, 1])], "motor_target")
    graph = helper.make_graph(
        nodes,
        "bello_actor",
        [
            helper.make_tensor_value_info(
                "observation", TensorProto.FLOAT, [1, observation_size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "action", TensorProto.FLOAT, [1, len(indices)]
            ),
            helper.make_tensor_value_info(
                "motor_target", TensorProto.FLOAT, [1, len(default)]
            ),
        ],
        initializer=constants,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bello_mujoco"
    )
    model.ir_version = 9
    helper.set_model_props(
        model, {"bello": json.dumps(metadata, allow_nan=False, separators=(",", ":"))}
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def export_checkpoint(config: ExportConfig) -> Path:
    from bello_mujoco.repository import (
        capture_repository_commit,
        require_checkpoint_output_directory,
        require_repository_commit,
    )
    from bello_mujoco.training.checkpoint import load_checkpoint_metadata
    from bello_mujoco.training.tracking import to_builtin

    if config.checkpoint_path is None or config.output_path is None:
        raise ValueError("checkpoint_path and output_path are required")
    output = config.output_path.expanduser().resolve()
    if output.suffix != ".zip":
        raise ValueError("output_path must name a .zip deployment bundle")
    require_checkpoint_output_directory(output.parent)
    if output.exists():
        raise FileExistsError(output)
    raw_metadata = json.loads(
        (config.checkpoint_path.expanduser().resolve() / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if raw_metadata.get("task") == "sonic_transfer":
        return export_sonic_checkpoint(config, output)
    if raw_metadata.get("task") == "supermimic":
        from bello_mujoco.supermimic.deployment import export_supermimic_checkpoint

        return export_supermimic_checkpoint(config, output)
    git_commit = capture_repository_commit()
    saved = load_checkpoint_metadata(config.checkpoint_path)
    if saved.task == "locomotion":
        from bello_mujoco.locomotion_runtime import load_locomotion_policy_runtime

        runtime = load_locomotion_policy_runtime(
            config.checkpoint_path, use_viewer_model=False
        )
    else:
        from bello_mujoco.motion.runtime import load_motion_policy_runtime

        runtime = load_motion_policy_runtime(
            config.checkpoint_path, use_viewer_model=False
        )
    env = runtime.env
    model = env.model
    metadata = {
        "git_commit": git_commit,
        "task": saved.task,
        "robot": model.robot.asset,
        "observation_size": saved.obs_size,
        "observation_frame": to_builtin(saved.actor_observation_layout.components),
        "observation_stack_size": saved.ppo.policy.obs_stack_size,
        "observation_max_abs": saved.ppo.rollout_observation_max_abs,
        "coordinates": model.actuator_coordinate_names,
        "actuators": model.actuator_names,
        "policy_indices": model.policy_ctrl_indices.tolist(),
        "default_position": model.default_ctrl.tolist(),
        "kp": model.mj_model.actuator_gainprm[:, 0].tolist(),
        "kd": (-model.mj_model.actuator_biasprm[:, 2]).tolist(),
        "control_timestep": env.control_timestep,
        "joint_from_motor": np.linalg.inv(model.actuator_coordinate_transform).tolist(),
        "default_joint_position": model.default_joint_target.tolist(),
        "joint_names": model.actuator_joint_names,
        "joint_limits": model.target_limits.joint.tolist(),
        "position_step_limits": model.target_limits.position_step.tolist(),
        "motor_from_joint": model.actuator_coordinate_transform.tolist(),
        "motor_limits": model.mj_model.actuator_ctrlrange.tolist(),
        "environment": to_builtin(env.config),
        "action_clip": env.action_clip,
        "hardware_imu_frame": model.robot.default_imu_body_name,
    }
    if saved.task == "locomotion":
        from bello_mujoco.sim import hardware_imu_to_policy_rotation

        metadata["command"] = to_builtin(env.command_config)
        metadata["policy_from_hardware"] = hardware_imu_to_policy_rotation(
            model
        ).tolist()
        metadata["left_axis"] = (-env._locomotion_right_axis()[:2]).tolist()
    else:
        metadata["end_effector_names"] = env.end_effector_names
    effort_scale = (
        np.asarray(env.beyondmimic_action_scale)
        if saved.task == "motion_tracking" and env.config.formulation == "beyondmimic"
        else None
    )
    graph = actor_onnx(
        runtime.policy,
        runtime.normalization.observation,
        saved.ppo,
        default_ctrl=model.default_ctrl,
        ctrl_limits=model.mj_model.actuator_ctrlrange,
        policy_indices=model.policy_ctrl_indices,
        action_clip=env.action_clip,
        metadata=metadata,
        effort_scale=effort_scale,
    )
    content = deployment_bundle(
        graph.SerializeToString(),
        physics=physics_mjcf(model),
        motion=env.motion if saved.task == "motion_tracking" else None,
    )
    require_repository_commit(git_commit)
    _publish_policy(output, content)
    return output


def sonic_actor_onnx(policy, *, model, env, metadata: dict):
    """Export the frozen encoder, FSQ, merged LoRA decoder, and Mini mapping."""

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from bello_mujoco.envs.actions import logical_action_scale

    nodes, constants = [], []

    def constant(name, value, *, dtype=np.float32):
        constants.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, inputs, output_name, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output_name], **attributes))
        return output_name

    actor = policy.actor
    config = actor.config
    step = actor.per_step_observation_size
    history = config.history_steps
    target_size = actor.adapter.target_size
    reference = op(
        "Gather",
        [
            "observation",
            constant(
                "reference_indices",
                np.arange((history - 1) * step, (history - 1) * step + 640),
                dtype=np.int64,
            ),
        ],
        "future_reference",
        axis=1,
    )

    def history_indices(offset, width):
        return np.concatenate(
            [
                np.arange(index * step + offset, index * step + offset + width)
                for index in range(history)
            ]
        )

    gravity_offset = config.reference_size
    angular_offset = gravity_offset + 3
    position_offset = angular_offset + 3
    velocity_offset = position_offset + target_size
    action_offset = velocity_offset + target_size

    def gather_history(name, offset, width):
        return op(
            "Gather",
            [
                "observation",
                constant(
                    f"{name}_indices",
                    history_indices(offset, width),
                    dtype=np.int64,
                ),
            ],
            name,
            axis=1,
        )

    angular = gather_history("angular_history", angular_offset, 3)
    gravity = gather_history("gravity_history", gravity_offset, 3)
    source_map = np.zeros((target_size, config.source_action_size), dtype=np.float32)
    for target, source in enumerate(actor.adapter.target_to_source_indices):
        source_map[target, source] = actor.adapter.signs[target]

    def mapped_history(name, offset):
        target = gather_history(f"target_{name}", offset, target_size)
        shaped = op(
            "Reshape",
            [
                target,
                constant(
                    f"{name}_target_shape", [history, target_size], dtype=np.int64
                ),
            ],
            f"{name}_target_history",
        )
        source = op(
            "MatMul",
            [shaped, constant(f"{name}_source_map", source_map)],
            f"{name}_source_history",
        )
        return op(
            "Reshape",
            [source, constant(f"{name}_source_shape", [1, -1], dtype=np.int64)],
            f"{name}_source_flat",
        )

    position = mapped_history("position", position_offset)
    velocity = mapped_history("velocity", velocity_offset)
    actions = mapped_history("actions", action_offset)
    proprioception = op(
        "Concat",
        [angular, position, velocity, actions, gravity],
        "proprioception",
        axis=1,
    )

    def mlp(value, network, prefix):
        for index, layer in enumerate(network.layers):
            value = op(
                "Gemm",
                [
                    value,
                    constant(
                        f"{prefix}_weight_{index}", np.asarray(layer.merged_weight()).T
                    ),
                    constant(f"{prefix}_bias_{index}", layer.bias),
                ],
                f"{prefix}_linear_{index}",
            )
            if index != len(network.layers) - 1:
                sigmoid = op("Sigmoid", [value], f"{prefix}_sigmoid_{index}")
                value = op("Mul", [value, sigmoid], f"{prefix}_silu_{index}")
        return value

    latent = mlp(reference, actor.encoder, "encoder")
    half_level = (config.fsq_levels - 1) * (1.0 - 1.0e-3) / 2.0
    fsq_offset = 0.5 if config.fsq_levels % 2 == 0 else 0.0
    fsq_shift = np.arctanh(fsq_offset / half_level) if fsq_offset else 0.0
    shifted = op("Add", [latent, constant("fsq_shift", fsq_shift)], "fsq_shifted")
    bounded = op("Tanh", [shifted], "fsq_tanh")
    bounded = op("Mul", [bounded, constant("fsq_half_level", half_level)], "fsq_scaled")
    bounded = op("Sub", [bounded, constant("fsq_offset", fsq_offset)], "fsq_bounded")
    rounded = op("Round", [bounded], "fsq_rounded")
    tokens = op(
        "Div",
        [rounded, constant("fsq_denominator", config.fsq_levels // 2)],
        "tokens",
    )
    source_action = mlp(
        op("Concat", [tokens, proprioception], "decoder_input", axis=1),
        actor.decoder,
        "decoder",
    )
    mapped_action = op(
        "Gather",
        [
            source_action,
            constant(
                "target_action_indices",
                actor.adapter.target_to_source_indices,
                dtype=np.int64,
            ),
        ],
        "mapped_action",
        axis=1,
    )
    action = op(
        "Mul",
        [mapped_action, constant("target_action_signs", actor.adapter.signs)],
        "action",
    )
    applied_action = op(
        "Clip",
        [
            action,
            constant("action_low", -env.action_clip),
            constant("action_high", env.action_clip),
        ],
        "applied_action",
    )
    logical_delta = op(
        "Mul",
        [
            applied_action,
            constant(
                "logical_action_scale",
                logical_action_scale(model, env.actuator_config),
            ),
        ],
        "logical_delta",
    )
    logical_target = op(
        "Add",
        [logical_delta, constant("default_joint_target", model.default_joint_target)],
        "logical_target",
    )
    requested = op(
        "MatMul",
        [
            logical_target,
            constant("motor_from_joint_graph", model.actuator_coordinate_transform.T),
        ],
        "requested_motor_target",
    )
    limits = np.asarray(model.mj_model.actuator_ctrlrange)
    lower = op(
        "Max",
        [requested, constant("motor_low_graph", limits[:, 0])],
        "motor_lower_bounded",
    )
    op("Min", [lower, constant("motor_high_graph", limits[:, 1])], "motor_target")

    graph = helper.make_graph(
        nodes,
        "bello_sonic_actor",
        [
            helper.make_tensor_value_info(
                "observation", TensorProto.FLOAT, [1, actor.observation_size]
            )
        ],
        [
            helper.make_tensor_value_info(
                "action", TensorProto.FLOAT, [1, target_size]
            ),
            helper.make_tensor_value_info(
                "motor_target", TensorProto.FLOAT, [1, target_size]
            ),
        ],
        initializer=constants,
    )
    onnx_model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="bello_mujoco",
    )
    onnx_model.ir_version = 9
    helper.set_model_props(
        onnx_model,
        {"bello": json.dumps(metadata, allow_nan=False, separators=(",", ":"))},
    )
    onnx.checker.check_model(onnx_model, full_check=True)
    return onnx_model


def export_sonic_checkpoint(config: ExportConfig, output: Path) -> Path:
    import mujoco

    from bello_mujoco.repository import (
        capture_repository_commit,
        require_repository_commit,
    )
    from bello_mujoco.sonic.runtime import load_sonic_policy_runtime
    from bello_mujoco.training.tracking import to_builtin

    if config.motion_dataset is None or not config.reference_clips:
        raise ValueError("SONIC export requires motion_dataset and reference_clips")
    if len(set(config.reference_clips)) != len(config.reference_clips) or any(
        not name or Path(name).name != name or Path(name).suffix
        for name in config.reference_clips
    ):
        raise ValueError("SONIC reference_clips must be unique plain clip stems")
    dataset = config.motion_dataset.expanduser().resolve()
    paths = tuple(dataset / f"{name}.npz" for name in config.reference_clips)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SONIC export clips are missing: {missing}")
    runtime = load_sonic_policy_runtime(
        config.checkpoint_path,
        paths,
        evaluation_names=(),
        use_viewer_model=False,
    )
    git_commit = capture_repository_commit()
    env, model = runtime.env, runtime.env.model
    kinematics = mujoco.MjData(model.mj_model)
    mujoco.mj_forward(model.mj_model, kinematics)
    root_id = mujoco.mj_name2id(
        model.mj_model, mujoco.mjtObj.mjOBJ_BODY, model.robot.root_body_name
    )
    imu_id = mujoco.mj_name2id(
        model.mj_model, mujoco.mjtObj.mjOBJ_BODY, model.robot.default_imu_body_name
    )
    if root_id < 0 or imu_id < 0:
        raise ValueError("SONIC deployment root or hardware IMU frame is missing")
    root_conjugate = np.asarray(kinematics.xquat[root_id]).copy()
    root_conjugate[1:] *= -1.0
    root_from_hardware_imu = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(
        root_from_hardware_imu,
        root_conjugate,
        np.asarray(kinematics.xquat[imu_id]),
    )
    metadata = {
        "git_commit": git_commit,
        "task": "sonic_transfer",
        "robot": model.robot.asset,
        "observation_size": runtime.policy.actor.observation_size,
        "observation_frame": to_builtin(env.observation_layout.components),
        "observation_stack_size": runtime.run_config.config.policy.obs_stack_size,
        "observation_max_abs": runtime.run_config.config.rollout_observation_max_abs,
        "coordinates": model.actuator_coordinate_names,
        "actuators": model.actuator_names,
        "policy_indices": model.policy_ctrl_indices.tolist(),
        "default_position": model.default_ctrl.tolist(),
        "kp": model.mj_model.actuator_gainprm[:, 0].tolist(),
        "kd": (-model.mj_model.actuator_biasprm[:, 2]).tolist(),
        "control_timestep": env.control_timestep,
        "joint_from_motor": np.linalg.inv(model.actuator_coordinate_transform).tolist(),
        "default_joint_position": model.default_joint_target.tolist(),
        "joint_names": model.actuator_joint_names,
        "joint_limits": model.target_limits.joint.tolist(),
        "position_step_limits": model.target_limits.position_step.tolist(),
        "motor_from_joint": model.actuator_coordinate_transform.tolist(),
        "motor_limits": model.mj_model.actuator_ctrlrange.tolist(),
        "environment": to_builtin(env.config),
        "action_clip": env.action_clip,
        "hardware_imu_frame": model.robot.default_imu_body_name,
        "root_from_hardware_imu_xyzw": np.roll(root_from_hardware_imu, -1).tolist(),
        "reference_clips": list(config.reference_clips),
        "future_frames": env.observation_assembler.future_frames,
        "future_dt": env.observation_assembler.future_dt,
        "source_joint_indices": env.embodiment.target_to_source_indices.tolist(),
        "source_neutral": env.embodiment.source_neutral.tolist(),
        "source_signs": env.embodiment.signs.tolist(),
        "pretrained_revision": runtime.metadata["pretrained_revision"],
        "pretrained_sha256": runtime.metadata["pretrained_sha256"],
        "dataset_artifact": runtime.metadata["dataset_artifact"],
    }
    graph = sonic_actor_onnx(
        runtime.policy,
        model=model,
        env=env,
        metadata=metadata,
    )
    content = multi_clip_deployment_bundle(
        graph.SerializeToString(),
        physics=physics_mjcf(model),
        clips=tuple(zip(config.reference_clips, paths, strict=True)),
    )
    require_repository_commit(git_commit)
    _publish_policy(output, content)
    return output


def multi_clip_deployment_bundle(
    policy: bytes,
    *,
    physics: bytes,
    clips: tuple[tuple[str, Path], ...],
) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("policy.onnx", policy)
        archive.writestr("physics.xml", physics)
        archive.writestr(
            "clips.json",
            json.dumps({"clips": [name for name, _ in clips]}, sort_keys=True),
        )
        for name, path in clips:
            archive.writestr(f"clips/{name}.npz", path.read_bytes())
    return buffer.getvalue()


def physics_mjcf(model) -> bytes:
    """Snapshot the runtime physics and home state without external assets."""
    import mujoco

    from bello_mujoco.models.serialization import save_model_xml

    if model.mj_model.nmesh or model.mj_model.ntex:
        raise ValueError("deployment physics must be loaded without visual assets")
    with tempfile.TemporaryDirectory(prefix="bello-physics-") as directory:
        # Restore the matching topology: mj_saveLastXML uses the last XML loaded.
        mujoco.MjModel.from_xml_path(str(model.xml_path))
        tree = save_model_xml(Path(directory) / "physics.xml", model.mj_model)
    root = tree.getroot()
    for attribute in ("meshdir", "texturedir", "assetdir"):
        root.find("compiler").attrib.pop(attribute, None)
    if any("file" in element.attrib for element in root.iter()):
        raise ValueError("deployment physics must not reference external files")
    key = root.find("keyframe/key[@name='home']")
    if key is None:
        raise ValueError("deployment physics is missing its home keyframe")
    for attribute, values in (
        ("qpos", model.default_qpos),
        ("qvel", model.default_qvel),
        ("ctrl", model.default_ctrl),
    ):
        key.set(attribute, " ".join(f"{value:.17g}" for value in values))
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # No asset dictionary or source directory: the bundle must stand alone.
    mujoco.MjModel.from_xml_string(content.decode("utf-8"))
    return content


def deployment_bundle(policy: bytes, *, physics: bytes, motion=None) -> bytes:
    from bello_mujoco.motion.clip import save_motion_clip

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("policy.onnx", policy)
        archive.writestr("physics.xml", physics)
        if motion is not None:
            clip = BytesIO()
            save_motion_clip(clip, motion)
            archive.writestr("motion.npz", clip.getvalue())
    return buffer.getvalue()


def _publish_policy(output: Path, content: bytes) -> None:
    """Publish a complete file without replacing an existing policy."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", dir=output.parent
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        # A hard link publishes atomically and fails if the destination exists.
        os.link(stream.name, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
