import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("status, expected", [(5, 0), (1, 1), (2, 2), (4, 4)])
def test_isolated_runner_skips_unselected_files_but_propagates_errors(
    tmp_path, status, expected
):
    uv = tmp_path / "uv"
    uv.write_text(
        f'#!/bin/sh\necho invoked-$3\n[ "$3" = first.py ] && exit {status}\nexit 0\n'
    )
    uv.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).parents[1] / "scripts/test.sh"),
            "first.py",
            "second.py",
        ],
        env=os.environ | {"PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected
    assert ("invoked-second.py" in result.stdout) == (status == 5)
