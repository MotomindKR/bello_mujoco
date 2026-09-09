from __future__ import annotations

import os
from pathlib import Path
import shutil
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
            "guestPortRange": [50051, 50053],
            "hostPortRange": [50051, 50053],
            "hostIP": "127.0.0.1",
            "proto": "tcp",
        },
        {"guestPort": 50054, "proto": "tcp", "ignore": True},
        {
            "guestPort": 50055,
            "hostPort": 50055,
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
        "NETWORKING_REVISION": "native-visualizer-v1",
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
    assert "{{.Param.NETWORKING_REVISION}}" in provision
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
    assert "require_current_networking_revision" in helper
    assert '{{index .Config.Param "NETWORKING_REVISION"}}' in helper
    assert "uses obsolete networking; delete and recreate it" in helper
    assert "{{(index .Config.Mounts 0).Location}}|" in helper
    assert "{{(index .Config.Mounts 0).MountPoint}}" in helper
    assert "submodule_status=$(git" in helper
    assert "grep -Eq '^[^ ]'" in helper
    assert "cargo build --locked -p bello-hardwared -p bello-state-estimatord" in helper
    assert "rustc --version && cargo --version" in helper


def test_legacy_networking_blocks_operations_but_allows_status_and_delete(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (repo / ".git").mkdir()
    helper = scripts / "lima_vcan.sh"
    shutil.copy2(_LIMA_HELPER, helper)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "limactl.log"
    fake_lima = fake_bin / "limactl"
    fake_lima.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "$BELLO_TEST_LIMA_LOG"
if [ "$1" = list ]; then
  case "$*" in
    *NETWORKING_REVISION*) exit 0 ;;
    *ENVIRONMENT*) printf '%s\\n' bello-mujoco-vcan-v1 ;;
    *Mounts*) printf '%s|%s\\n' "$BELLO_TEST_REPO" /workspace/bello_mujoco ;;
    *Status*) printf '%s\\n' Running ;;
    *Name*) printf '%s\\n' bello-vcan ;;
    *) printf '%s\\n' 'bello-vcan Running' ;;
  esac
fi
exit 0
"""
    )
    fake_lima.chmod(0o755)
    for command, body in (
        ("git", "#!/bin/sh\nexit 0\n"),
        ("uname", "#!/bin/sh\nprintf '%s\\n' Darwin\n"),
    ):
        executable = fake_bin / command
        executable.write_text(body)
        executable.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "BELLO_TEST_LIMA_LOG": str(log),
        "BELLO_TEST_REPO": str(repo),
    }
    for arguments in (["start"], ["bootstrap"], ["run", "true"]):
        operation = subprocess.run(
            ["bash", helper, *arguments],
            capture_output=True,
            text=True,
            env=environment,
        )
        assert operation.returncode != 0
        assert "uses obsolete networking" in operation.stderr

    status = subprocess.run(
        ["bash", helper, "status"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "uses obsolete networking" in status.stderr

    subprocess.run(["bash", helper, "delete"], check=True, env=environment)
    calls = log.read_text().splitlines()
    assert "start bello-vcan" not in calls
    assert "stop bello-vcan" in calls
    assert "delete bello-vcan" in calls


def test_lima_nix_shell_keeps_platform_artifacts_off_the_host_mount() -> None:
    flake = (_ROOT / "flake.nix").read_text()

    assert flake.count("uv python install --no-bin") == 2
    assert "lima-vcan = pkgs.mkShell" in flake
    assert 'export UV_PROJECT_ENVIRONMENT="$lima_state/venv"' in flake
    assert 'export UV_PYTHON_INSTALL_DIR="$lima_state/python"' in flake
    assert 'export WARP_CACHE_PATH="$lima_cache/warp"' in flake
    assert 'export CARGO_TARGET_DIR="$lima_cache/cargo-target"' in flake
    assert "export JAX_PLATFORMS=cpu" in flake
    assert "host.lima.internal,127.0.0.1,localhost" in flake


def test_macos_visualizer_shell_uses_host_local_cargo_cache() -> None:
    flake = (_ROOT / "flake.nix").read_text()

    assert "macos-visualizer = pkgs.mkShell" in flake
    assert 'export CARGO_HOME="$macos_visualizer_cache/cargo-home"' in flake
    assert 'export CARGO_TARGET_DIR="$macos_visualizer_cache/cargo-target"' in flake
    assert "deps/bello-stack/components/bello-visualizer/Cargo.toml" in flake


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
