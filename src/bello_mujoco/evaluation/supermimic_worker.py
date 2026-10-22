"""Isolated CPU worker for full-corpus native SuperMimic evaluation."""

import json
import os
import sys


def main():
    request = json.load(sys.stdin)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        assigned = available[request["index"] :: request["workers"]]
        if assigned:
            os.sched_setaffinity(0, assigned)
    from bello_mujoco.evaluation.software import configure_software_rendering

    configure_software_rendering()
    from bello_mujoco.evaluation.supermimic import render_supermimic_checkpoint

    render_supermimic_checkpoint(
        request["checkpoint"],
        request["dataset"],
        tuple(request["selected"]),
        request["output"],
        video_clips=tuple(request["videos"]),
    )


if __name__ == "__main__":
    main()
