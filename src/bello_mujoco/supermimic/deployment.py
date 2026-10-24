"""Actor-only ONNX graphs with an explicit Mini observation and goal codec."""

import json

import jax.numpy as jnp
import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bello_mujoco.deployment import (
    _publish_policy,
    multi_clip_deployment_bundle,
    physics_mjcf,
)
from bello_mujoco.repository import capture_repository_commit, require_repository_commit
from bello_mujoco.supermimic.checkpoint import load_actor
from bello_mujoco.training.tracking import to_builtin


def actor_onnx(actor, observation_norm, goal_norm, *, env, metadata):
    nodes, constants = [], []

    def constant(name, value, dtype=np.float32):
        constants.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], **attributes))
        return output

    history = env.codec.config.history_frames
    sample_size, goal_size = env.codec.sample_size, env.codec.goal_size
    history_size = history * sample_size
    history_input = op(
        "Gather",
        ["observation", constant("history_indices", np.arange(history_size), np.int64)],
        "history",
        axis=1,
    )
    goal = op(
        "Gather",
        [
            "observation",
            constant(
                "goal_indices",
                np.arange(history_size, history_size + goal_size),
                np.int64,
            ),
        ],
        "goal",
        axis=1,
    )
    normalized_history = op(
        "Div",
        [
            op(
                "Sub",
                [
                    history_input,
                    constant(
                        "observation_mean",
                        np.tile(np.asarray(observation_norm.mean), history),
                    ),
                ],
                "history_centered",
            ),
            constant(
                "observation_std", np.tile(np.asarray(observation_norm.std()), history)
            ),
        ],
        "history_normalized",
    )
    normalized_goal = op(
        "Div",
        [
            op("Sub", [goal, constant("goal_mean", goal_norm.mean)], "goal_centered"),
            constant("goal_std", goal_norm.std()),
        ],
        "goal_normalized",
    )

    def mlp(value, network, prefix):
        if network.norms:
            raise ValueError(
                "exported actors must not contain hidden layer normalization"
            )
        mixed = network.matmul_precision == "bfloat16"

        def round_activation(value, name):
            reduced = op("Cast", [value], name + "_bf16", to=TensorProto.BFLOAT16)
            return op("Cast", [reduced], name, to=TensorProto.FLOAT)

        def parameter(value):
            return value.astype(jnp.bfloat16).astype(jnp.float32) if mixed else value

        for index, layer in enumerate(network.layers):
            if mixed:
                value = round_activation(value, f"{prefix}_input_{index}")
            value = op(
                "Gemm",
                [
                    value,
                    constant(
                        f"{prefix}_weight_{index}",
                        np.asarray(parameter(layer.weight)).T,
                    ),
                    constant(f"{prefix}_bias_{index}", parameter(layer.bias)),
                ],
                f"{prefix}_linear_{index}",
            )
            if mixed:
                value = round_activation(value, f"{prefix}_rounded_linear_{index}")
            if index < len(network.layers) - 1:
                if network.activation != "elu":
                    raise ValueError(
                        "SuperMimic actor export requires ELU hidden activations"
                    )
                value = op("Elu", [value], f"{prefix}_activation_{index}", alpha=1.0)
                if mixed:
                    value = round_activation(
                        value, f"{prefix}_rounded_activation_{index}"
                    )
        return value

    latent = op("Tanh", [mlp(normalized_goal, actor.encoder, "encoder")], "latent")
    decoder = op("Concat", [latent, normalized_history], "decoder_input", axis=1)
    action = op("Identity", [mlp(decoder, actor.decoder, "decoder")], "action")
    applied = op(
        "Clip",
        [
            action,
            constant("action_low", -env.action_clip),
            constant("action_high", env.action_clip),
        ],
        "applied_action",
    )
    logical = op(
        "Add",
        [
            op(
                "Mul",
                [applied, constant("action_scale", env.mapping.scale)],
                "joint_delta",
            ),
            constant("neutral_joint", env.mapping.neutral_joint),
        ],
        "logical_target",
    )
    op(
        "MatMul",
        [
            logical,
            constant("motor_from_joint", np.asarray(env.mapping.motor_from_joint).T),
        ],
        "motor_target",
    )
    n = len(env.mapping.joint_names)
    graph = helper.make_graph(
        nodes,
        "bello_supermimic_actor",
        [
            helper.make_tensor_value_info(
                "observation", TensorProto.FLOAT, [1, history_size + goal_size]
            )
        ],
        [
            helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, n]),
            helper.make_tensor_value_info("motor_target", TensorProto.FLOAT, [1, n]),
        ],
        initializer=constants,
    )
    result = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bello_mujoco"
    )
    result.ir_version = 9
    helper.set_model_props(
        result, {"bello": json.dumps(metadata, allow_nan=False, separators=(",", ":"))}
    )
    onnx.checker.check_model(result, full_check=True)
    return result


def export_supermimic_checkpoint(config, output):
    if config.motion_dataset is None or not config.reference_clips:
        raise ValueError(
            "SuperMimic export requires motion_dataset and reference_clips"
        )
    env, actor, obs_norm, goal_norm, run_config = load_actor(
        config.checkpoint_path,
        dataset_directory=config.motion_dataset,
        reference_clips=config.reference_clips,
    )
    commit = capture_repository_commit()
    model, codec = env.model, env.codec
    data = mujoco.MjData(model.mj_model)
    data.qpos[:] = model.default_qpos
    mujoco.mj_forward(model.mj_model, data)
    root = model.mj_model.body(env.config.anchor_body_name).id
    imu = model.mj_model.body(model.robot.default_imu_body_name).id
    inverse = data.xquat[root].copy()
    inverse[1:] *= -1
    root_from_imu = np.empty(4)
    mujoco.mju_mulQuat(root_from_imu, inverse, data.xquat[imu])
    history_size = codec.sample_size * codec.config.history_frames
    metadata = {
        "git_commit": commit,
        "task": "supermimic",
        "robot": model.robot.asset,
        "observation_size": history_size + codec.goal_size,
        "observation_frame": [
            ["proprioceptive_history", history_size],
            ["future_reference", codec.goal_size],
        ],
        "observation_stack_size": 1,
        "observation_max_abs": 10000.0,
        "sample_frame": to_builtin(env.observation_layout.components),
        "history_frames": codec.config.history_frames,
        "future_frames": codec.config.future_frames,
        "reference_orientation": codec.config.reference_orientation,
        "goal_joint_velocity_scale": codec.config.goal_joint_velocity_scale,
        "rotation_6d_layout": "first_two_columns_row_major",
        "reference_root_body": env.config.anchor_body_name,
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
        "root_from_hardware_imu_xyzw": np.roll(root_from_imu, -1).tolist(),
        "reference_clips": list(config.reference_clips),
        "dataset_artifact": run_config.dataset.artifact,
    }
    graph = actor_onnx(actor, obs_norm, goal_norm, env=env, metadata=metadata)
    content = multi_clip_deployment_bundle(
        graph.SerializeToString(),
        physics=physics_mjcf(model),
        clips=tuple(
            (name, run_config.dataset.directory / f"{name}.npz")
            for name in config.reference_clips
        ),
    )
    require_repository_commit(commit)
    _publish_policy(output, content)
    return output
