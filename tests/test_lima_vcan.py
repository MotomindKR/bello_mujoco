from __future__ import annotations

from pathlib import Path
import stat
import subprocess
import tomllib

import yaml


_ROOT = Path(__file__).resolve().parents[1]
_LIMA_TEMPLATE = _ROOT / "lima" / "bello-vcan.yaml"
_LIMA_HELPER = _ROOT / "scripts" / "lima_vcan.sh"


def test_lima_template_is_isolated_and_provisions_vcan() -> None:
    template = yaml.safe_load(_LIMA_TEMPLATE.read_text())

    assert template["minimumLimaVersion"] == "2.0.0"
    assert template["base"] == ["template:_images/ubuntu-24.04"]
    assert template["vmType"] == "vz"
    assert template["arch"] == "aarch64"
    assert template["mountType"] == "virtiofs"
    assert template["containerd"] == {"system": False, "user": False}
    assert template["portForwards"] == [
        {
            "guestPortRange": [50051, 50055],
            "hostPortRange": [50051, 50055],
            "hostIP": "127.0.0.1",
            "proto": "tcp",
        },
        {"guestPortRange": [1, 65535], "proto": "any", "ignore": True},
    ]
    assert template["mounts"] == [
        {
            "location": "{{.Param.REPO_DIR}}",
            "mountPoint": "/workspace/bello_mujoco",
            "writable": True,
        }
    ]
    assert template["param"] == {
        "ENVIRONMENT": "bello-mujoco-vcan-v1",
        "REPO_DIR": "/REPLACE/WHEN/CREATING/INSTANCE",
    }

    system_provisions = [
        entry["script"] for entry in template["provision"] if entry["mode"] == "system"
    ]
    assert len(system_provisions) == 1
    provision = system_provisions[0]
    assert "export HOME=/root" in provision
    assert "export USER=root" in provision
    assert "{{.Param.ENVIRONMENT}}" in provision
    assert "linux-modules-extra-$(uname -r)" in provision
    assert "https://nixos.org/nix/install" in provision
    assert '"$emulator_setup" start' in provision
    assert "vcan0" in template["probes"][0]["script"]
    assert "vcan1" in template["probes"][0]["script"]


def test_lima_helper_is_valid_bash_and_documents_all_actions() -> None:
    assert _LIMA_HELPER.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", _LIMA_HELPER], check=True)
    result = subprocess.run(
        ["bash", _LIMA_HELPER, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for action in (
        "start",
        "bootstrap",
        "shell",
        "run CMD...",
        "doctor",
        "status",
        "stop",
        "delete",
    ):
        assert action in result.stdout

    helper = _LIMA_HELPER.read_text()
    assert "require_owned_instance" in helper
    assert '{{index .Config.Param "ENVIRONMENT"}}' in helper
    assert "require_matching_checkout" in helper
    assert "{{(index .Config.Mounts 0).Location}}|" in helper
    assert "{{(index .Config.Mounts 0).MountPoint}}" in helper
    assert "submodule_status=$(git" in helper
    assert "grep -Eq '^[^ ]'" in helper


def test_lima_nix_shell_keeps_platform_artifacts_off_the_host_mount() -> None:
    flake = (_ROOT / "flake.nix").read_text()

    assert "lima-vcan = pkgs.mkShell" in flake
    assert 'export UV_PROJECT_ENVIRONMENT="$lima_state/venv"' in flake
    assert 'export UV_PYTHON_INSTALL_DIR="$lima_state/python"' in flake
    assert 'export WARP_CACHE_PATH="$lima_cache/warp"' in flake
    assert 'export CARGO_TARGET_DIR="$lima_cache/cargo-target"' in flake
    assert "export JAX_PLATFORMS=cpu" in flake


def test_linux_arm_lock_uses_cpu_jax_while_x86_training_keeps_cuda() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    dependencies = project["dependencies"]

    assert any(
        dependency.startswith("jax>=")
        and "sys_platform == 'linux' and platform_machine == 'aarch64'" in dependency
        for dependency in dependencies
    )
    assert any(
        dependency.startswith("jax[cuda13]>=")
        and "sys_platform == 'linux' and platform_machine == 'x86_64'" in dependency
        for dependency in dependencies
    )
    assert not any(
        dependency.startswith("jax[cuda13]>=")
        and "platform_machine == 'aarch64'" in dependency
        for dependency in dependencies
    )
