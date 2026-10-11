"""Opt-in parity against the compiled Rust consumer, without a robot or GPU."""

import json
from io import BytesIO
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import pytest

from bello_mujoco.deployment import actor_onnx, deployment_bundle
from bello_mujoco.envs.actions import bounded_motor_target
from bello_mujoco.policy import ActorCritic, ActorCriticConfig
from bello_mujoco.training.ppo_types import ObservationNormState
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.normalization import normalize_observation


def floating_joint_physics(joint_names):
    """Minimal model with an explicit generalized-coordinate joint order."""
    root = ET.Element("mujoco")
    base = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    ET.SubElement(base, "freejoint")
    ET.SubElement(base, "geom", type="sphere", size=".1", mass="1")
    for name in joint_names:
        body = ET.SubElement(base, "body")
        ET.SubElement(body, "joint", name=name)
        ET.SubElement(body, "geom", type="sphere", size=".1", mass="1")
    return ET.tostring(root)


@pytest.fixture
def rust_policy(tmp_path):
    executable = os.environ.get("BELLO_POLICY_BIN")
    if not executable:
        pytest.skip("set BELLO_POLICY_BIN and ORT_DYLIB_PATH for Rust parity")
    executable = str(Path(executable).resolve(strict=True))

    def git(path, *args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=Runtime Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "protocol.file.allow=always",
                *args,
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    source = tmp_path / "source"
    dependency = tmp_path / "dependency"
    sonic_dependency = tmp_path / "sonic-dependency"
    for path in (source, dependency, sonic_dependency):
        path.mkdir()
        git(path, "init")
        git(path, "commit", "--allow-empty", "-m", "Synthetic fixture")
    git(source, "submodule", "add", str(dependency), "deps/bello-stack")
    git(source, "submodule", "add", str(sonic_dependency), "deps/gear_sonic")
    git(source, "commit", "-am", "Pin synthetic dependency")
    commit = git(source, "rev-parse", "HEAD")
    config = PPOConfig(policy=ActorCriticConfig(hidden_dims=(8,), activation="swish"))
    actor = ActorCritic(3, 2, config.policy, key=jax.random.PRNGKey(7))
    norm = ObservationNormState(jnp.array([0.1, 0.2, 0.3]), jnp.ones(3), jnp.array(3.0))
    default = jnp.array([0.2, -0.3, 0.4])
    limits = jnp.array([[-1.0, 2.0], [-3.0, 1.0], [-1.0, 1.0]])
    indices = jnp.array([0, 2])
    metadata = {
        "git_commit": commit,
        "task": "locomotion",
        "robot": "synthetic",
        "observation_size": 3,
        "observation_frame": [["synthetic", 3]],
        "observation_stack_size": 1,
        "observation_max_abs": 100.0,
        "coordinates": ["first", "held", "last"],
        "policy_indices": [0, 2],
        "default_position": default.tolist(),
        "motor_limits": limits.tolist(),
        "joint_limits": limits.tolist(),
        "joint_from_motor": jnp.eye(3).tolist(),
        "position_step_limits": [2.0] * 3,
        "kp": [2.0] * 3,
        "kd": [0.1] * 3,
        "control_timestep": 0.02,
    }

    def run(
        observations=(),
        *,
        metadata_updates=None,
        content=None,
        check=False,
        omit_physics=False,
        telemetry=False,
        hardware_config=None,
        clip=None,
    ):
        if content is None:
            graph = actor_onnx(
                actor,
                norm,
                config,
                default_ctrl=default,
                ctrl_limits=limits,
                policy_indices=indices,
                action_clip=1.0,
                metadata=metadata | (metadata_updates or {}),
            )
            content = deployment_bundle(graph.SerializeToString(), physics=b"<mujoco/>")
        if omit_physics:
            buffer = BytesIO()
            with ZipFile(BytesIO(content)) as original, ZipFile(buffer, "w") as archive:
                for name in original.namelist():
                    if name != "physics.xml":
                        archive.writestr(name, original.read(name))
            content = buffer.getvalue()
        bundle = tmp_path / "deployment.zip"
        bundle.write_bytes(content)
        hardware_args = []
        if hardware_config is not None:
            config_path = tmp_path / "hardware.yaml"
            config_path.write_text(json.dumps(hardware_config))
            hardware_args = ["--hardware-config", str(config_path)]
        return subprocess.run(
            [executable, "--policy", str(bundle), "--source", str(source)]
            + (["--clip", clip] if clip is not None else [])
            + (["--check"] if check else [])
            + (["--telemetry"] if telemetry else [])
            + hardware_args,
            # No hardware writes are authorized by the read-only preflight mode.
            # Telemetry mode only consumes stdin; it never connects or arms hardware.
            input="".join(json.dumps(row) + "\n" for row in observations),
            text=True,
            capture_output=True,
            timeout=30,
        )

    def expected(observation):
        action = actor.action_mean(
            normalize_observation(jnp.asarray(observation), norm, config)
        )
        target = bounded_motor_target(action, default, limits, indices, 1.0)
        return action, target

    return run, expected, source
