"""Optional offline warm-start for SONIC's low-rank dynamics adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bello_mujoco.motion.sonic_dataset import (
    SONIC_CRITIC_FUTURE_OFFSETS,
    SonicReferenceBank,
)
from bello_mujoco.policy import ActorCritic, SonicActor
from bello_mujoco.sonic_policy import (
    BELLO_TO_G1_VECTOR_BASIS,
    SONIC_BELLO_ACTION_SIZE,
    SONIC_G1_ACTION_SCALE,
    SONIC_G1_DEFAULT_JOINT_POSITION,
    SONIC_PROPRIO_HISTORY_FRAMES,
    SONIC_ROBOT_INPUT_SIZE,
    SONIC_TOKEN_FLAT_SIZE,
    SonicReleaseWeights,
    semantic_g1_from_bello_joint_map,
    sonic_fsq,
)


@dataclass(frozen=True)
class SonicPretrainingConfig:
    enabled: bool = True
    updates: int = 2_000
    batch_size: int = 512
    validation_batch_size: int = 2_048
    learning_rate: float = 1.0e-4
    max_gradient_norm: float = 1.0
    validation_clip_fraction: float = 0.05
    startup_history_fraction: float = 0.5
    action_huber_delta: float = 0.5
    activation_consistency_weight: float = 0.05
    temporal_consistency_weight: float = 0.1
    target_limit_weight: float = 0.05
    parameter_anchor_weight: float = 1.0e-6
    log_every_updates: int = 100
    seed: int = 17

    def __post_init__(self) -> None:
        for name in (
            "updates",
            "batch_size",
            "validation_batch_size",
            "log_every_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"pretraining.{name} must be a positive integer")
        for name in (
            "learning_rate",
            "max_gradient_norm",
            "action_huber_delta",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"pretraining.{name} must be positive and finite")
        if not 0.0 < self.validation_clip_fraction < 1.0:
            raise ValueError("pretraining.validation_clip_fraction must be in (0, 1)")
        if not 0.0 <= self.startup_history_fraction <= 1.0:
            raise ValueError("pretraining.startup_history_fraction must be in [0, 1]")
        for name in (
            "activation_consistency_weight",
            "temporal_consistency_weight",
            "target_limit_weight",
            "parameter_anchor_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"pretraining.{name} must be nonnegative and finite")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("pretraining.seed must be a nonnegative integer")


class SonicPretrainingBatch(NamedTuple):
    observation: np.ndarray
    next_observation: np.ndarray
    g1_proprioception: np.ndarray
    action_target: np.ndarray
    next_action_target: np.ndarray


class SonicOfflineBatchBuilder:
    """Sample synchronized ideal-tracking tuples without simulator rollouts."""

    def __init__(
        self,
        bank: SonicReferenceBank,
        *,
        bello_default_joint_position: np.ndarray,
        bello_action_scale: np.ndarray,
        validation_clip_fraction: float,
        startup_history_fraction: float,
        seed: int,
    ) -> None:
        self.bank = bank
        self.bello_default = _vector(
            bello_default_joint_position,
            SONIC_BELLO_ACTION_SIZE,
            "Bello default joint position",
        )
        self.bello_action_scale = _positive_vector(
            bello_action_scale,
            SONIC_BELLO_ACTION_SIZE,
            "Bello action scale",
        )
        self.g1_from_bello_joint = semantic_g1_from_bello_joint_map(bank.calibration)
        if not 0.0 <= startup_history_fraction <= 1.0:
            raise ValueError("startup history fraction must be in [0, 1]")
        self.startup_history_fraction = float(startup_history_fraction)
        if bank.qpos.shape[1:] != (7 + SONIC_BELLO_ACTION_SIZE,):
            raise ValueError(
                "SONIC pretraining requires packed Bello policy qpos order"
            )
        if bank.qvel.shape[1:] != (6 + SONIC_BELLO_ACTION_SIZE,):
            raise ValueError(
                "SONIC pretraining requires packed Bello policy qvel order"
            )
        if bank.g1_joint_pos.shape[1:] != (len(SONIC_G1_ACTION_SCALE),):
            raise ValueError("SONIC pretraining G1 joint shape is invalid")
        qpos_index = {name: index for index, name in enumerate(bank.qpos_joint_names)}
        if set(qpos_index) != set(bank.joint_names):
            raise ValueError("SONIC pretraining Bello joint orders are inconsistent")
        self._policy_qpos_index = np.asarray(
            [qpos_index[name] for name in bank.joint_names], dtype=np.int32
        )
        self._rng = np.random.default_rng(seed)
        train, validation = _partition_clips(
            bank,
            validation_clip_fraction=validation_clip_fraction,
        )
        self._clip_indices = {"train": train, "validation": validation}
        self._clip_probabilities = {
            split: _restricted_clip_probabilities(bank, indices)
            for split, indices in self._clip_indices.items()
        }

    def batch(
        self,
        split: Literal["train", "validation"],
        batch_size: int,
    ) -> SonicPretrainingBatch:
        if split not in self._clip_indices:
            raise ValueError(f"invalid SONIC pretraining split {split!r}")
        clip_candidates = self._clip_indices[split]
        clip_indices = self._rng.choice(
            clip_candidates,
            size=batch_size,
            replace=True,
            p=self._clip_probabilities[split],
        )
        lengths = np.asarray(self.bank.lengths)[clip_indices]
        minimum_frame = 10
        maximum_frame = lengths - max(SONIC_CRITIC_FUTURE_OFFSETS) - 2
        if np.any(maximum_frame < minimum_frame):
            raise ValueError("SONIC pretraining clip does not cover its history/future")
        frames = minimum_frame + np.floor(
            self._rng.random(batch_size) * (maximum_frame - minimum_frame + 1)
        ).astype(np.int32)
        absolute = np.asarray(self.bank.offsets)[clip_indices] + frames
        startup = self._rng.random(batch_size) < self.startup_history_fraction
        startup_age = np.where(
            startup,
            self._rng.integers(0, SONIC_PROPRIO_HISTORY_FRAMES, size=batch_size),
            SONIC_PROPRIO_HISTORY_FRAMES,
        ).astype(np.int32)
        return SonicPretrainingBatch(
            observation=self._observation(absolute, startup_age),
            next_observation=self._observation(
                absolute + 1,
                np.minimum(startup_age + 1, SONIC_PROPRIO_HISTORY_FRAMES),
            ),
            g1_proprioception=self._g1_proprioception(absolute, startup_age),
            action_target=self._action_target(absolute + 1),
            next_action_target=self._action_target(absolute + 2),
        )

    def _observation(self, absolute: np.ndarray, startup_age: np.ndarray) -> np.ndarray:
        future = absolute[:, None] + np.asarray(SONIC_CRITIC_FUTURE_OFFSETS)[None, :]
        history, reset_slots = _startup_history(absolute, startup_age)
        current_quaternion = np.asarray(self.bank.qpos)[absolute, 3:7]
        current_rotation = _quaternion_rotation_matrix(current_quaternion)
        reference_rotation = _quaternion_rotation_matrix(
            np.asarray(self.bank.qpos)[future, 3:7]
        )
        relative_rotation = np.einsum(
            "ij,bjk,bfkl,lm->bfim",
            BELLO_TO_G1_VECTOR_BASIS,
            np.swapaxes(current_rotation, -1, -2),
            reference_rotation,
            BELLO_TO_G1_VECTOR_BASIS.T,
        )
        orientation = relative_rotation[..., :2].reshape(
            len(absolute), len(future[0]), 6
        )
        bello_joint_position = np.asarray(self.bank.qpos)[
            future[:, :, None],
            7 + self._policy_qpos_index[None, None, :],
        ]
        bello_joint_velocity = np.asarray(self.bank.qvel)[
            future[:, :, None],
            6 + self._policy_qpos_index[None, None, :],
        ]
        g1_joint_position = SONIC_G1_DEFAULT_JOINT_POSITION + np.einsum(
            "ij,bfj->bfi",
            self.g1_from_bello_joint,
            bello_joint_position - self.bello_default,
        )
        g1_joint_velocity = np.einsum(
            "ij,bfj->bfi", self.g1_from_bello_joint, bello_joint_velocity
        )
        command = np.concatenate(
            (
                g1_joint_position.reshape(len(absolute), -1),
                g1_joint_velocity.reshape(len(absolute), -1),
            ),
            axis=-1,
        ).reshape(len(absolute), len(future[0]), -1)
        robot_frames = np.concatenate((command, orientation), axis=-1).reshape(
            len(absolute), -1
        )
        proprioception = self._bello_proprioception(history, reset_slots)
        return np.concatenate((robot_frames, proprioception), axis=-1).astype(
            np.float32
        )

    def _bello_proprioception(
        self, history: np.ndarray, reset_slots: np.ndarray
    ) -> np.ndarray:
        qpos = np.asarray(self.bank.qpos)[history]
        qvel = np.asarray(self.bank.qvel)[history]
        rotation = _quaternion_rotation_matrix(qpos[..., 3:7])
        gravity = np.einsum(
            "bhij,j->bhi",
            np.swapaxes(rotation, -1, -2),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        )
        angular = np.einsum(
            "bhij,bhj->bhi",
            np.swapaxes(rotation, -1, -2),
            qvel[..., 3:6],
        )
        joint_position = qpos[..., 7 + self._policy_qpos_index] - self.bello_default
        joint_velocity = qvel[..., 6 + self._policy_qpos_index]
        current_qpos = np.asarray(self.bank.qpos)[
            history[:, :, None],
            7 + self._policy_qpos_index[None, None, :],
        ]
        previous_action = (current_qpos - self.bello_default) / self.bello_action_scale
        joint_velocity = np.where(reset_slots[:, :, None], 0.0, joint_velocity)
        previous_action = np.where(reset_slots[:, :, None], 0.0, previous_action)
        return _term_major_history(
            gravity,
            angular,
            joint_position,
            joint_velocity,
            previous_action,
        )

    def _g1_proprioception(
        self, absolute: np.ndarray, startup_age: np.ndarray
    ) -> np.ndarray:
        history, reset_slots = _startup_history(absolute, startup_age)
        qpos = np.asarray(self.bank.qpos)[history]
        qvel = np.asarray(self.bank.qvel)[history]
        rotation = _quaternion_rotation_matrix(qpos[..., 3:7])
        bello_gravity = np.einsum(
            "bhij,j->bhi",
            np.swapaxes(rotation, -1, -2),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        )
        bello_angular = np.einsum(
            "bhij,bhj->bhi",
            np.swapaxes(rotation, -1, -2),
            qvel[..., 3:6],
        )
        gravity = np.einsum("ij,bhj->bhi", BELLO_TO_G1_VECTOR_BASIS, bello_gravity)
        angular = np.einsum("ij,bhj->bhi", BELLO_TO_G1_VECTOR_BASIS, bello_angular)
        g1_position = np.asarray(self.bank.g1_joint_pos)[history]
        previous_g1 = np.asarray(self.bank.g1_joint_pos)[history - 1]
        joint_position = g1_position - SONIC_G1_DEFAULT_JOINT_POSITION
        joint_velocity = (g1_position - previous_g1) * 50.0
        previous_action = (
            g1_position - SONIC_G1_DEFAULT_JOINT_POSITION
        ) / SONIC_G1_ACTION_SCALE
        joint_velocity = np.where(reset_slots[:, :, None], 0.0, joint_velocity)
        previous_action = np.where(reset_slots[:, :, None], 0.0, previous_action)
        return _term_major_history(
            gravity,
            angular,
            joint_position,
            joint_velocity,
            previous_action,
        )

    def _action_target(self, absolute: np.ndarray) -> np.ndarray:
        bello_delta = (
            np.asarray(self.bank.qpos)[
                absolute[:, None],
                7 + self._policy_qpos_index[None, :],
            ]
            - self.bello_default
        )
        return (
            np.einsum("ij,bj->bi", self.g1_from_bello_joint, bello_delta)
            / SONIC_G1_ACTION_SCALE
        ).astype(np.float32)


def pretrain_sonic_adapters(
    policy: ActorCritic,
    release: SonicReleaseWeights,
    builder: SonicOfflineBatchBuilder,
    config: SonicPretrainingConfig,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    progress: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[ActorCritic, tuple[tuple[int, dict[str, float]], ...]]:
    """Fit only actor LoRA residuals before constructing PPO state."""

    if not isinstance(policy.actor, SonicActor):
        raise TypeError("SONIC pretraining requires a SONIC actor")
    action_low = _vector(action_low, SONIC_BELLO_ACTION_SIZE, "action lower bound")
    action_high = _vector(action_high, SONIC_BELLO_ACTION_SIZE, "action upper bound")
    if np.any(action_low >= action_high):
        raise ValueError("SONIC pretraining action bounds are not ordered")
    trainable_filter = _adapter_pretraining_filter(policy.actor)
    parameters, static_actor = eqx.partition(policy.actor, trainable_filter)
    initial_parameters = jax.tree_util.tree_map(lambda value: value.copy(), parameters)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )
    optimizer_state = optimizer.init(parameters)
    release_first_weight = jnp.asarray(release.decoder_weights[0])
    release_first_bias = jnp.asarray(release.decoder_biases[0])
    low = jnp.asarray(action_low)
    high = jnp.asarray(action_high)

    def loss_fn(params, batch):
        actor = eqx.combine(params, static_actor)
        prediction = jax.vmap(actor)(batch.observation)
        next_prediction = jax.vmap(actor)(batch.next_observation)
        action_loss = jnp.mean(
            optax.huber_loss(
                prediction,
                batch.action_target,
                delta=config.action_huber_delta,
            )
        )
        temporal_loss = jnp.mean(
            optax.huber_loss(
                next_prediction - prediction,
                batch.next_action_target - batch.action_target,
                delta=config.action_huber_delta,
            )
        )
        tokens = jax.vmap(
            lambda observation: sonic_fsq(
                actor.robot_encoder(observation[:SONIC_ROBOT_INPUT_SIZE]).reshape(
                    (2, 32)
                )
            ).reshape((SONIC_TOKEN_FLAT_SIZE,))
        )(batch.observation)
        bello_proprioception = batch.observation[:, SONIC_ROBOT_INPUT_SIZE:]
        decoder = actor.decoder
        aligned_proprioception = bello_proprioception @ jax.lax.stop_gradient(
            decoder.g1_from_bello_proprio
        ).T + jax.lax.stop_gradient(decoder.g1_from_bello_proprio_offset)
        first_weight = jax.lax.stop_gradient(decoder.base_weights[0])
        student_first = (
            tokens @ first_weight[:, :SONIC_TOKEN_FLAT_SIZE].T
            + aligned_proprioception @ first_weight[:, SONIC_TOKEN_FLAT_SIZE:].T
            + decoder.lora_scale
            * (aligned_proprioception @ decoder.lora_a[0].T)
            @ decoder.lora_b[0].T
            + jax.lax.stop_gradient(decoder.base_biases[0])
        )
        teacher_first = (
            tokens @ release_first_weight[:, :SONIC_TOKEN_FLAT_SIZE].T
            + batch.g1_proprioception
            @ release_first_weight[:, SONIC_TOKEN_FLAT_SIZE:].T
            + release_first_bias
        )
        activation_loss = jnp.mean((student_first - teacher_first) ** 2)
        mapped_prediction = jax.vmap(decoder.bello_action)(prediction)
        limit_excess = jax.nn.relu(low - mapped_prediction) + jax.nn.relu(
            mapped_prediction - high
        )
        target_limit_loss = jnp.mean(limit_excess**2)
        parameter_anchor_loss = _parameter_mean_square_error(params, initial_parameters)
        loss = (
            action_loss
            + config.activation_consistency_weight * activation_loss
            + config.temporal_consistency_weight * temporal_loss
            + config.target_limit_weight * target_limit_loss
            + config.parameter_anchor_weight * parameter_anchor_loss
        )
        metrics = {
            "loss": loss,
            "action_huber": action_loss,
            "action_prediction_rms": jnp.sqrt(jnp.mean(prediction**2)),
            "action_target_rms": jnp.sqrt(jnp.mean(batch.action_target**2)),
            "action_error_rms": jnp.sqrt(
                jnp.mean((prediction - batch.action_target) ** 2)
            ),
            "activation_rms": jnp.sqrt(activation_loss),
            "temporal_huber": temporal_loss,
            "target_limit_loss": target_limit_loss,
            "target_outside_fraction": jnp.mean(
                ((prediction < low) | (prediction > high)).astype(jnp.float32)
            ),
            "parameter_anchor": parameter_anchor_loss,
        }
        return loss, metrics

    value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit
    def update(params, optimizer_state, batch):
        (_, metrics), gradients = value_and_grad(params, batch)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        return eqx.apply_updates(params, updates), optimizer_state, metrics

    records: list[tuple[int, dict[str, float]]] = []
    for update_index in range(1, config.updates + 1):
        batch = _device_batch(builder.batch("train", config.batch_size))
        parameters, optimizer_state, metrics = update(
            parameters, optimizer_state, batch
        )
        if update_index % config.log_every_updates == 0 or update_index == 1:
            record = _host_metrics(metrics)
            records.append((update_index, record))
            if progress is not None:
                progress(update_index, record)
    validation_batch = _device_batch(
        builder.batch("validation", config.validation_batch_size)
    )
    _, validation_metrics = loss_fn(parameters, validation_batch)
    validation_record = {
        f"validation/{name}": value
        for name, value in _host_metrics(validation_metrics).items()
    }
    validation_update = config.updates + 1
    records.append((validation_update, validation_record))
    if progress is not None:
        progress(validation_update, validation_record)
    actor = eqx.combine(parameters, static_actor)
    return eqx.tree_at(lambda value: value.actor, policy, actor), tuple(records)


def _adapter_pretraining_filter(actor: SonicActor):
    specification = jax.tree_util.tree_map(lambda _: False, actor)
    for selector in (
        lambda value: value.decoder.lora_a,
        lambda value: value.decoder.lora_b,
    ):
        specification = eqx.tree_at(
            selector,
            specification,
            tuple(True for _ in selector(actor)),
        )
    return specification


def _device_batch(batch: SonicPretrainingBatch) -> SonicPretrainingBatch:
    return SonicPretrainingBatch(*(jnp.asarray(value) for value in batch))


def _host_metrics(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value.block_until_ready())))
        for name, value in metrics.items()
    }


def _parameter_mean_square_error(parameters, initial_parameters) -> jax.Array:
    leaves = [
        jnp.mean((value - initial) ** 2)
        for value, initial in zip(
            jax.tree_util.tree_leaves(parameters),
            jax.tree_util.tree_leaves(initial_parameters),
            strict=True,
        )
    ]
    return sum(leaves, jnp.asarray(0.0, dtype=jnp.float32)) / len(leaves)


def _partition_clips(
    bank: SonicReferenceBank,
    *,
    validation_clip_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.arange(bank.valid_clip_count, dtype=np.int32)
    if len(valid) < 2:
        raise ValueError("SONIC pretraining requires at least two training clips")
    scores = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{bank.source_ids[index]}/{bank.sequence_ids[index]}".encode()
                ).digest()[:8],
                "big",
            )
            / float(2**64)
            for index in valid
        ]
    )
    validation = valid[scores < validation_clip_fraction]
    if len(validation) == 0:
        validation = valid[np.asarray((int(np.argmin(scores)),))]
    train = valid[~np.isin(valid, validation)]
    if len(train) == 0:
        raise ValueError("SONIC pretraining validation partition consumed every clip")
    return train, validation


def _restricted_clip_probabilities(
    bank: SonicReferenceBank, clip_indices: np.ndarray
) -> np.ndarray:
    probability = np.asarray(bank.sampling_probabilities, dtype=np.float64)[
        clip_indices
    ]
    total = float(np.sum(probability))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("SONIC pretraining clip probability is invalid")
    return probability / total


def _startup_history(
    absolute: np.ndarray, startup_age: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the history buffer seen during the first ten rollout frames."""

    absolute = np.asarray(absolute, dtype=np.int32)
    startup_age = np.asarray(startup_age, dtype=np.int32)
    if absolute.ndim != 1 or startup_age.shape != absolute.shape:
        raise ValueError("SONIC startup history indices must be equal-length vectors")
    if np.any((startup_age < 0) | (startup_age > SONIC_PROPRIO_HISTORY_FRAMES)):
        raise ValueError("SONIC startup history age must be in [0, 10]")
    nominal = (
        absolute[:, None]
        + np.arange(
            -SONIC_PROPRIO_HISTORY_FRAMES + 1,
            1,
            dtype=np.int32,
        )[None, :]
    )
    startup_frame = absolute - startup_age
    history = np.maximum(nominal, startup_frame[:, None])
    reset_slots = (startup_age < SONIC_PROPRIO_HISTORY_FRAMES)[:, None] & (
        history == startup_frame[:, None]
    )
    return history, reset_slots


def _term_major_history(
    gravity: np.ndarray,
    angular: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    batch_size = gravity.shape[0]
    return np.concatenate(
        tuple(
            value.reshape(batch_size, -1)
            for value in (
                gravity,
                angular,
                joint_position,
                joint_velocity,
                action,
            )
        ),
        axis=-1,
    ).astype(np.float32)


def _quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("SONIC pretraining encountered a zero quaternion")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result


def _positive_vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    result = _vector(value, size, label)
    if np.any(result <= 0.0):
        raise ValueError(f"{label} must be positive")
    return result
