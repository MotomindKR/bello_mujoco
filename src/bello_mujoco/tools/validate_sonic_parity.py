"""Numerically compare release checkpoint layers between PyTorch and JAX."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.sim import load_model
from bello_mujoco.sonic.checkpoint import (
    import_sonic_checkpoint,
    load_source_checkpoint,
)
from bello_mujoco.sonic.embodiment import MiniEmbodimentAdapter
from bello_mujoco.sonic.model import finite_scalar_quantize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    results = validate(args.checkpoint, args.revision, seed=args.seed, atol=args.atol)
    print(results)


def validate(
    checkpoint_path: Path,
    revision: str,
    *,
    seed: int = 0,
    atol: float = 2.0e-5,
) -> dict[str, float]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "parity validation requires --group reference-validation"
        ) from error
    model = load_model(
        ModelConfig(asset="bello_mini", actuator_mode="policy"),
        load_mjx=False,
    )
    adapter = MiniEmbodimentAdapter.from_model(model)
    _validate_pinned_source_order()
    imported = import_sonic_checkpoint(
        checkpoint_path,
        revision=revision,
        adapter=adapter,
        key=jax.random.PRNGKey(seed),
    )
    checkpoint = load_source_checkpoint(checkpoint_path)
    policy_state = checkpoint["policy_state_dict"]
    value_state = checkpoint["value_state_dict"]
    rng = np.random.default_rng(seed)

    def torch_mlp(value, state, prefix, indices):
        value = torch.from_numpy(value)
        for offset, index in enumerate(indices):
            value = functional.linear(
                value,
                state[f"{prefix}.{index}.weight"].float(),
                state[f"{prefix}.{index}.bias"].float(),
            )
            if offset != len(indices) - 1:
                value = functional.silu(value)
        return value.detach().numpy()

    reference = rng.standard_normal((3, 640), dtype=np.float32)
    encoder_torch = torch_mlp(
        reference,
        policy_state,
        "actor_module.encoders.g1.module",
        (0, 2, 4, 6, 8),
    )
    encoder_jax = np.asarray(jax.vmap(imported.policy.actor.encoder)(reference))
    _assert_close("encoder", encoder_jax, encoder_torch, atol)

    half_l = 31.0 * (1.0 - 1.0e-3) / 2.0
    bounded = torch.tanh(torch.from_numpy(encoder_torch) + np.arctanh(0.5 / half_l))
    fsq_torch = (torch.round(bounded * half_l - 0.5) / 16).numpy()
    fsq_jax = np.asarray(finite_scalar_quantize(jnp.asarray(encoder_jax), 32))
    _assert_close("quantizer", fsq_jax, fsq_torch, atol)

    decoder_input = rng.standard_normal((3, 994), dtype=np.float32)
    decoder_torch = torch_mlp(
        decoder_input,
        policy_state,
        "actor_module.decoders.g1_dyn.module",
        (0, 2, 4, 6, 8, 10, 12),
    )
    decoder_jax = np.asarray(jax.vmap(imported.policy.actor.decoder)(decoder_input))
    _assert_close("decoder_zero_lora", decoder_jax, decoder_torch, atol)

    critic_input_size = imported.policy.critic.network.input_size
    critic_input = rng.standard_normal((3, critic_input_size), dtype=np.float32)
    critic_torch = torch_mlp(
        critic_input,
        value_state,
        "critic_module.module",
        (0, 2, 4, 6, 8, 10, 12),
    )
    critic_jax = np.asarray(jax.vmap(imported.policy.critic.network)(critic_input))
    _assert_close("critic_zero_lora", critic_jax, critic_torch, atol)
    source_std = policy_state["std"].detach().cpu().float().numpy()
    expected_std = source_std[adapter.target_to_source_indices]
    imported_std = np.exp(np.asarray(imported.policy.log_std))
    _assert_close("exploration_adapter", imported_std, expected_std, atol)
    return {
        "encoder_max_abs": float(np.max(np.abs(encoder_jax - encoder_torch))),
        "quantizer_max_abs": float(np.max(np.abs(fsq_jax - fsq_torch))),
        "decoder_max_abs": float(np.max(np.abs(decoder_jax - decoder_torch))),
        "critic_max_abs": float(np.max(np.abs(critic_jax - critic_torch))),
        "exploration_adapter_max_abs": float(
            np.max(np.abs(imported_std - expected_std))
        ),
    }


def _assert_close(name: str, actual, expected, atol: float) -> None:
    if not np.allclose(actual, expected, rtol=1.0e-5, atol=atol):
        maximum = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name} parity failed: max_abs={maximum}, atol={atol}")


def _validate_pinned_source_order() -> None:
    from bello_mujoco.sonic.embodiment import G1_JOINT_NAMES

    expected = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    )
    if G1_JOINT_NAMES != expected:
        raise AssertionError(
            "SONIC source DOF order differs from the pinned Any2Any bridge"
        )


if __name__ == "__main__":
    main()
