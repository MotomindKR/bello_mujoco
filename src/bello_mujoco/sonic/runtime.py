"""Restore a self-contained SONIC actor and its recorded runtime contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import equinox as eqx
import jax

from bello_mujoco.configuration import dataclass_from_dict
from bello_mujoco.envs.sonic_transfer import BelloSonicTransferEnv
from bello_mujoco.repository import require_repository_commit
from bello_mujoco.sonic.model import (
    SonicActorCritic,
    SonicNetworkConfig,
    sonic_policy_template,
)
from bello_mujoco.sonic.corpus import MANIFEST, file_sha256
from bello_mujoco.training.sonic_transfer import SonicTransferTrainingRunConfig


@dataclass(frozen=True)
class SonicPolicyRuntime:
    env: BelloSonicTransferEnv
    policy: SonicActorCritic
    run_config: SonicTransferTrainingRunConfig
    metadata: dict


def load_sonic_policy_runtime(
    checkpoint_path: Path,
    motion_paths: tuple[Path, ...],
    *,
    evaluation_names: tuple[str, ...],
    use_viewer_model: bool,
    evaluation_input: str = "robot",
) -> SonicPolicyRuntime:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    metadata = _load_metadata(checkpoint_path)
    if metadata["dataset_manifest_sha256"] is not None:
        manifest_path = checkpoint_path / MANIFEST
        if file_sha256(manifest_path) != metadata["dataset_manifest_sha256"]:
            raise ValueError("checkpoint corpus manifest checksum differs")
        entries = {
            entry["file"]: entry
            for entry in json.loads(manifest_path.read_text())["clips"]
        }
        for path in motion_paths:
            if (
                path.name not in entries
                or file_sha256(path) != entries[path.name]["sha256"]
            ):
                raise ValueError(
                    f"selected reference differs from checkpoint corpus: {path.name}"
                )
    require_repository_commit(metadata["git_commit"])
    run_config = dataclass_from_dict(
        SonicTransferTrainingRunConfig,
        metadata["resolved_config"],
        strict=True,
        require_all_fields=True,
        path="resolved_config",
    )
    if run_config.config.normalize_observations:
        raise ValueError("SONIC deployment requires unnormalized actor observations")
    env = BelloSonicTransferEnv.load_corpus(
        run_config.asset_config,
        replace(
            run_config.env_config,
            model_randomization=replace(
                run_config.env_config.model_randomization, enabled=False
            ),
        ),
        motion_paths,
        evaluation_names=evaluation_names,
        warp=run_config.warp,
        batch_size=1,
        use_viewer_model=use_viewer_model,
        load_mjx=False,
        sampling_config=run_config.sampling_config,
        input_config=run_config.input_config,
        evaluation_input=evaluation_input,
    )
    template = sonic_policy_template(
        env.embodiment,
        key=jax.random.PRNGKey(run_config.config.seed),
        config=SonicNetworkConfig(smpl_enabled=run_config.input_config.smpl_enabled),
    )
    policy = eqx.tree_deserialise_leaves(checkpoint_path / "policy.eqx", template)
    if policy.actor.observation_size != (
        env.observation_size * run_config.config.policy.obs_stack_size
    ):
        raise ValueError("SONIC checkpoint actor layout does not match the corpus")
    return SonicPolicyRuntime(
        env=env,
        policy=policy,
        run_config=run_config,
        metadata=metadata,
    )


def _load_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid SONIC checkpoint metadata: {metadata_path}"
        ) from error
    if not isinstance(value, dict) or value.get("task") != "sonic_transfer":
        raise ValueError(f"checkpoint is not a SONIC transfer checkpoint: {path}")
    required = {
        "task",
        "update",
        "git_commit",
        "pretrained_revision",
        "pretrained_sha256",
        "dataset_artifact",
        "dataset_manifest_sha256",
        "clips",
        "resolved_config",
    }
    if set(value) != required:
        raise ValueError(
            "SONIC checkpoint metadata fields differ from the supported contract"
        )
    if not (path / "policy.eqx").is_file():
        raise FileNotFoundError(path / "policy.eqx")
    return value
