"""Compare first-party physical/nonblank LoC with a Git revision (no dependencies)."""

from collections import defaultdict
from pathlib import Path
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def category(name: str) -> str | None:
    path = Path(name)
    if path.name.endswith(("_pb2.py", "_pb2_grpc.py")):
        return None
    if path.suffix not in {".py", ".sh", ".yaml", ".toml", ".md", ".nix"}:
        return None
    return next(
        (
            prefix
            for prefix in ("src", "tests", "configs", "scripts", "docs")
            if path.parts[0] == prefix
        ),
        None,
    )


def count(revision: str | None) -> dict[str, list[int]]:
    names = (
        git("ls-tree", "-r", "--name-only", revision)
        if revision
        else git("ls-files", "--cached", "--others", "--exclude-standard")
    )
    totals = defaultdict(lambda: [0, 0])
    for name in sorted(set(names.splitlines())):
        group = category(name)
        if group is None or (revision is None and not Path(name).is_file()):
            continue
        source = (
            git("show", f"{revision}:{name}") if revision else Path(name).read_text()
        )
        lines = source.splitlines()
        totals[group][0] += len(lines)
        totals[group][1] += sum(bool(line.strip()) for line in lines)
    return totals


if __name__ == "__main__":
    baseline = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    before, after = count(baseline), count(None)
    print(
        f"Baseline: {baseline}; excludes vendored/generated code and output artifacts"
    )
    print("Category  Physical before -> now (delta)     Nonblank before -> now")
    for group in sorted(before.keys() | after.keys()):
        old, new = before[group], after[group]
        print(
            f"{group:9} {old[0]:7} -> {new[0]:7} ({new[0] - old[0]:+7})"
            f"     {old[1]:7} -> {new[1]:7}"
        )
