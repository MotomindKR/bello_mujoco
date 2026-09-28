"""Content-addressed corpus manifests and verified, immutable W&B artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

MANIFEST = "sonic-corpus.json"
PINNED_ARTIFACT = re.compile(r"[^/]+/[^/]+/[^/:]+:(?:v[0-9]+|[0-9a-f]{32,})\Z")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_corpus(
    directory: Path,
    splits: dict[str, dict[str, str]],
    *,
    provenance: dict | None = None,
) -> dict:
    """Record each clip's original sequence identity to prevent split leakage."""
    paths = sorted(directory.glob("*.npz"))
    if not paths or set(splits) != {path.stem for path in paths}:
        raise ValueError("splits must describe exactly every .npz clip in the corpus")
    entries = []
    for path in paths:
        if set(splits[path.stem]) != {"split", "source_sequence"}:
            raise ValueError(
                "split records must contain only split and source_sequence"
            )
        with np.load(path, allow_pickle=False) as arrays:
            size = sum(arrays[name].nbytes for name in arrays.files)
            for name in arrays.files:
                value = arrays[name]
                if (
                    np.issubdtype(value.dtype, np.number)
                    and not np.isfinite(value).all()
                ):
                    raise ValueError(f"nonfinite corpus array: {path.name}/{name}")
        entries.append(
            {
                "file": path.name,
                "sha256": file_sha256(path),
                "array_bytes": size,
                **splits[path.stem],
            }
        )
    document = {"schema": 1, "clips": entries}
    if provenance is not None:
        document["provenance"] = provenance
    _validate_document(document)
    destination = directory / MANIFEST
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return document


def _validate_document(document: dict) -> None:
    if (
        set(document) not in ({"schema", "clips"}, {"schema", "clips", "provenance"})
        or document["schema"] != 1
        or not document["clips"]
    ):
        raise ValueError("unsupported or empty SONIC corpus manifest")
    if "provenance" in document and not isinstance(document["provenance"], dict):
        raise ValueError("corpus provenance must be an object")
    groups = {}
    files = set()
    for entry in document["clips"]:
        if set(entry) != {"file", "sha256", "array_bytes", "split", "source_sequence"}:
            raise ValueError(
                "corpus entries require file, sha256, array_bytes, split, source_sequence"
            )
        name = entry["file"]
        if Path(name).name != name or Path(name).suffix != ".npz" or name in files:
            raise ValueError("corpus filenames must be unique, flat .npz paths")
        files.add(name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or not isinstance(entry["array_bytes"], int)
            or entry["array_bytes"] <= 0
        ):
            raise ValueError("invalid corpus checksum or size")
        split, source = entry["split"], entry["source_sequence"]
        if (
            split not in {"train", "evaluation"}
            or not isinstance(source, str)
            or not source.strip()
        ):
            raise ValueError("each clip requires a split and original source_sequence")
        if source in groups and groups[source] != split:
            raise ValueError(f"source sequence appears in both splits: {source}")
        groups[source] = split
    if "train" not in groups.values() or "evaluation" not in groups.values():
        raise ValueError(
            "corpus requires separate training and evaluation source sequences"
        )


def verify_corpus(directory: Path, evaluation_clips: tuple[str, ...]) -> str:
    document = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    _validate_document(document)
    expected_files = {entry["file"] for entry in document["clips"]} | {MANIFEST}
    if {path.name for path in directory.iterdir()} != expected_files:
        raise ValueError("corpus files differ from its manifest")
    held_out = {
        Path(entry["file"]).stem
        for entry in document["clips"]
        if entry["split"] == "evaluation"
    }
    if (
        len(set(evaluation_clips)) != len(evaluation_clips)
        or set(evaluation_clips) != held_out
    ):
        raise ValueError(
            "evaluation_clips must exactly match the immutable corpus split"
        )
    for entry in document["clips"]:
        path = directory / entry["file"]
        if path.is_symlink() or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"corpus content changed: {path.name}")
    return file_sha256(directory / MANIFEST)


def verify_artifact(
    directory: Path, reference: str, evaluation_clips: tuple[str, ...], *, run=None
) -> str:
    if PINNED_ARTIFACT.fullmatch(reference) is None:
        raise ValueError("dataset requires ENTITY/PROJECT/NAME:vN or immutable digest")
    import wandb

    artifact = (
        run.use_artifact(reference, type="dataset")
        if run is not None
        else wandb.Api().artifact(reference, type="dataset")
    )
    if any(entry.ref is not None for entry in artifact.manifest.entries.values()):
        raise ValueError(
            "SONIC dataset must contain bytes, not mutable external references"
        )
    artifact.verify(root=str(directory))
    return verify_corpus(directory, evaluation_clips)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("--splits", type=Path, required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("directory", type=Path)
    upload.add_argument("--entity", required=True)
    upload.add_argument("--project", required=True)
    upload.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        document = prepare_corpus(args.directory, json.loads(args.splits.read_text()))
        print(
            json.dumps(
                {
                    "clips": len(document["clips"]),
                    "array_bytes": sum(
                        entry["array_bytes"] for entry in document["clips"]
                    ),
                }
            )
        )
        return
    document = json.loads((args.directory / MANIFEST).read_text())
    evaluation = tuple(
        Path(entry["file"]).stem
        for entry in document["clips"]
        if entry["split"] == "evaluation"
    )
    digest = verify_corpus(args.directory, evaluation)
    import wandb

    visibility = wandb.Api()._service_api.execute_graphql(
        "query($entity: String!, $project: String!) { project(name: $project, entityName: $entity) { access } }",
        {"entity": args.entity, "project": args.project},
    )
    if (visibility.get("project") or {}).get("access") != "PRIVATE":
        raise ValueError("corpus upload requires an existing private W&B project")
    with wandb.init(
        entity=args.entity, project=args.project, job_type="sonic-corpus-upload"
    ) as run:
        artifact = wandb.Artifact(
            args.name, type="dataset", metadata={"manifest_sha256": digest}
        )
        artifact.add_dir(str(args.directory))
        logged = run.log_artifact(artifact).wait()
        print(
            json.dumps({"artifact": logged.qualified_name, "manifest_sha256": digest})
        )


if __name__ == "__main__":
    main()
