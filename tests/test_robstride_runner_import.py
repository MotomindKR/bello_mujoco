from __future__ import annotations

import os
import subprocess
import sys


def test_runner_import_does_not_load_jax_or_mjx() -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from omegaconf import OmegaConf; "
            "from bello_mujoco.hydra_config import structured_config; "
            "from bello_mujoco.tools.robstride_runner import RobstrideRunnerConfig; "
            "structured_config(RobstrideRunnerConfig, OmegaConf.create({})); "
            "unexpected = sorted(name for name in sys.modules "
            "if name == 'jax' or name.startswith('jax.') "
            "or name == 'mujoco.mjx' or name.startswith('mujoco.mjx.')); "
            "assert not unexpected, unexpected",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
