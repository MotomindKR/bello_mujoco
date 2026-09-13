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
from bello_mujoco.training.ppo import ObservationNormState, PPOConfig


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: Path | None = None
    output_path: Path | None = None


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
        "joint_limits": model.mj_model.jnt_range[model.actuator_joint_ids].tolist(),
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
