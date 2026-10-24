"""CPU-only checkpoint queue worker; configure the process before importing JAX."""

from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
import traceback


def numeric_metrics(document, prefix=""):
    result = {}
    for name, value in document.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            result.update(numeric_metrics(value, key))
        elif isinstance(value, (int, float)):
            result[key] = value
    return result


def evaluate_job(job, request, root, tracking_run):
    from bello_mujoco.configuration import dataclass_from_dict
    from bello_mujoco.evaluation.checkpoint_config import CheckpointEvaluationConfig
    from bello_mujoco.evaluation.world_model_audit import (
        WorldModelAuditConfig,
        audit_world_model,
    )
    from bello_mujoco.sonic.corpus import MANIFEST, file_sha256
    from bello_mujoco.supermimic.checkpoint import checkpoint_metadata
    from bello_mujoco.training.checkpoint_evaluation import write_json_atomic

    config = dataclass_from_dict(
        CheckpointEvaluationConfig, request["evaluation"], strict=True
    )
    metadata = checkpoint_metadata(job["checkpoint"])
    identity = {
        key: metadata[key]
        for key in (
            "git_commit",
            "dataset_manifest_sha256",
            "actor_sha256",
            "dynamics_sha256",
        )
    }
    identity["metadata_sha256"] = file_sha256(Path(job["checkpoint"]) / "metadata.json")
    if metadata["update"] != job["update"] or identity != job["identity"]:
        raise ValueError("queued checkpoint identity changed")
    dataset = Path(request["dataset"])
    manifest = json.loads((dataset / MANIFEST).read_text())
    selected = tuple(
        Path(e["file"]).stem for e in manifest["clips"] if e["split"] == "train"
    )
    if any(w.clip not in selected for w in config.windows):
        raise ValueError("checkpoint windows must select training clips")
    output = root / "results" / f"{job['update']:09d}"
    model_report = audit_world_model(
        WorldModelAuditConfig(
            Path(job["checkpoint"]), dataset, output / "windows.json", config.windows
        )
    )
    rows = [window["tracking"] for window in model_report["windows"].values()]
    tracking = {
        "window_count": len(rows),
        "split": "training",
        "measures_held_out_generalization": False,
        "weighting": "equal_per_window",
        "start": "explicit_window",
        "failure_fraction": mean(float(row["terminal"]) for row in rows),
        **{
            f"mean_{key}": mean(row[key] for row in rows)
            for key in rows[0]
            if key not in ("terminal", "timeout")
            and isinstance(rows[0][key], (int, float))
        },
    }
    summary = {
        "update": job["update"],
        "identity": job["identity"],
        "tracking": tracking,
        "window_report": "windows.json",
    }
    metrics = numeric_metrics(tracking, "tracking")
    for name, window in model_report["windows"].items():
        metrics.update(numeric_metrics(window["horizons"], f"world_model/{name}"))
        metrics.update(numeric_metrics(window["tracking"], f"window_tracking/{name}"))
    write_json_atomic(output / "summary.json", summary)
    if tracking_run is not None:
        tracking_run.log(
            {"checkpoint_update": job["update"], **metrics}, step=job["update"]
        )
        import wandb

        artifact = wandb.Artifact(
            request["run_name"] + "-checkpoint-evaluations",
            type="evaluation",
            metadata={"checkpoint_update": job["update"], **job["identity"]},
        )
        artifact.add_dir(str(output))
        tracking_run.log_artifact(artifact).wait()
    print(
        json.dumps({"checkpoint_evaluated": job["update"], "tracking": tracking}),
        flush=True,
    )


def run_queue(root, request, tracking_run):
    from bello_mujoco.training.checkpoint_evaluation import write_json_atomic

    completed = []
    while True:
        pending = [
            p for p in sorted((root / "jobs").glob("*.json")) if p.name not in completed
        ]
        if pending:
            path = pending[0]
            evaluate_job(json.loads(path.read_text()), request, root, tracking_run)
            completed.append(path.name)
            write_json_atomic(
                root / "status.json", {"completed": completed, "finished": False}
            )
        elif (root / "finish").exists():
            write_json_atomic(
                root / "status.json", {"completed": completed, "finished": True}
            )
            return
        else:
            time.sleep(1)


def main():
    root = Path(sys.argv[1]).resolve()
    request = json.loads((root / "request.json").read_text())
    os.environ.update(
        JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        count = request["evaluation"]["cpu_cores"]
        os.sched_setaffinity(0, available[-count:])
    os.nice(10)
    from bello_mujoco.configuration import dataclass_from_dict
    from bello_mujoco.training.tracking import (
        WandbConfig,
        close_run,
        create_wandb_run,
        log_config,
    )
    from bello_mujoco.training.checkpoint_evaluation import write_json_atomic

    tracking = dataclass_from_dict(WandbConfig, request["wandb"], strict=True)
    tracking = replace(
        tracking,
        job_type="checkpoint-evaluation",
        run_name=request["run_name"] + "-checkpoint-evaluation",
    )
    run = None
    succeeded = False
    try:
        run = create_wandb_run(tracking)
        log_config(
            run,
            {
                "parent_run_url": request["parent_run_url"],
                "evaluation": request["evaluation"],
            },
        )
        write_json_atomic(
            root / "worker.json",
            {
                "wandb_url": run.url if run is not None else None,
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
            },
        )
        run_queue(root, request, run)
        succeeded = True
    except BaseException:
        write_json_atomic(root / "error.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
