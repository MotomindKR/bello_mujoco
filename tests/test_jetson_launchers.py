"""Launcher argument handling must not reach robot control."""

from pathlib import Path
import subprocess

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.mark.parametrize("script", ["run_jetson_policy.sh", "run_jetson_motion.sh"])
def test_launcher_help_is_read_only(script):
    result = subprocess.run(
        ["bash", str(SCRIPTS / script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "read-only preflight" in result.stdout


@pytest.mark.parametrize(
    "script,args",
    [
        ("run_jetson_motion.sh", []),
        ("run_jetson_motion.sh", ["--arm"]),
        ("run_jetson_motion.sh", ["../policy", "--arm"]),
        ("run_jetson_policy.sh", ["--motion"]),
        ("run_jetson_policy.sh", ["--motion", "../policy", "--arm"]),
        ("run_jetson_policy.sh", ["--arm", "--arm"]),
    ],
)
def test_invalid_arguments_fail_before_accessing_deployment(script, args):
    result = subprocess.run(
        ["bash", str(SCRIPTS / script), *args], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "Required deployment file missing" not in result.stderr
