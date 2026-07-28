"""Repository identity and exact-source paths."""

from __future__ import annotations

import re
import subprocess
import warnings
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def capture_repository_commit() -> str:
    """Capture a clean HEAD represented by a local origin-tracking branch."""

    root = repository_root()
    _require_clean_worktree(root)
    commit = _head_commit(root)
    remote_refs = _git_output(
        root,
        "for-each-ref",
        "--format=%(refname)",
        f"--contains={commit}",
        "refs/remotes/origin",
    ).splitlines()
    if not any(
        ref.startswith("refs/remotes/origin/") and ref != "refs/remotes/origin/HEAD"
        for ref in remote_refs
    ):
        raise RuntimeError(
            f"HEAD {commit} is not contained by a local origin-tracking "
            "branch; fetch or push the training commit before creating checkpoints"
        )
    return commit


def require_repository_commit(
    expected: str,
    *,
    allow_mismatch: bool = False,
) -> None:
    """Validate a clean checkout and, by default, its exact commit."""

    validate_repository_commit(expected)
    root = repository_root()
    _require_clean_worktree(root)
    current_commit = _head_commit(root)
    if current_commit != expected:
        if allow_mismatch:
            warnings.warn(
                "Git commit mismatch override enabled: "
                f"checkpoint={expected}, current={current_commit}. Compatibility "
                "is not guaranteed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        raise ValueError(
            "checkpoint source commit does not match this checkout: "
            f"checkpoint={expected}, current={current_commit}. Create a separate "
            "checkout at the checkpoint commit and run uv sync --frozen there"
        )


def require_checkpoint_output_directory(path: Path) -> Path:
    """Require checkpoint output outside the source tree or under an ignore rule."""

    root = repository_root()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved

    result = _run_git(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        (relative / ".bello-checkpoint-output").as_posix(),
        text=True,
    )
    if result.returncode == 0:
        return resolved
    if result.returncode == 1:
        raise ValueError(
            "checkpoint output inside the source repository must be Git-ignored: "
            f"{relative.as_posix()}"
        )
    _raise_git_error(result, ("check-ignore", "--quiet", "--no-index"))
    raise AssertionError("unreachable")


def validate_repository_commit(commit: str) -> None:
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("git_commit must contain exactly 40 lowercase hex digits")


def require_tracked_repository_path(path: Path) -> Path:
    """Return a repository-relative path that exists in the current commit."""

    root = repository_root()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"checkpoint input must be inside the repository: {path}"
        ) from error
    relative_path = relative.as_posix()
    try:
        _git_output(root, "cat-file", "-e", f"HEAD:{relative_path}")
    except RuntimeError as error:
        raise ValueError(
            f"checkpoint input must be tracked in HEAD: {relative_path}"
        ) from error
    return Path(relative_path)


def resolve_repository_path(path: Path) -> Path:
    """Resolve a repository-relative checkpoint path independently of the CWD."""

    relative = Path(path)
    if relative.is_absolute():
        raise ValueError(f"checkpoint path must be repository-relative: {path}")
    root = repository_root()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"checkpoint path escapes the repository: {path}") from error
    return resolved


def repository_root() -> Path:
    package_root = Path(__file__).resolve().parent
    root = Path(_git_output(package_root, "rev-parse", "--show-toplevel")).resolve()
    if package_root != root / "src" / "bello_mujoco":
        raise RuntimeError(
            "exact-source operations require bello_mujoco to run from the source "
            f"checkout at {root}"
        )
    return root


def _require_clean_worktree(root: Path) -> None:
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if not status:
        return
    changed = status.splitlines()
    preview = ", ".join(changed[:5])
    if len(changed) > 5:
        preview += ", ..."
    raise RuntimeError(
        "exact-source checkpoint operations require a clean Git worktree; "
        f"commit or remove these changes first: {preview}"
    )


def _head_commit(root: Path) -> str:
    commit = _git_output(root, "rev-parse", "HEAD")
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError(f"Git returned an invalid HEAD commit: {commit!r}")
    return commit


def _git_output(root: Path, *arguments: str) -> str:
    result = _run_git(root, *arguments, text=True)
    if result.returncode != 0:
        _raise_git_error(result, arguments)
    assert isinstance(result.stdout, str)
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = _run_git(root, *arguments, text=False)
    if result.returncode != 0:
        stderr = result.stderr
        stdout = result.stdout
        assert isinstance(stderr, bytes)
        assert isinstance(stdout, bytes)
        detail = stderr.decode(errors="replace").strip()
        if not detail:
            detail = stdout.decode(errors="replace").strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
        )
    assert isinstance(result.stdout, bytes)
    return result.stdout


def _run_git(
    root: Path,
    *arguments: str,
    text: bool,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=text,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required for exact-source operations") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Git exact-source operation timed out") from error


def _raise_git_error(
    result: subprocess.CompletedProcess,
    arguments: tuple[str, ...],
) -> None:
    assert isinstance(result.stderr, str)
    assert isinstance(result.stdout, str)
    detail = result.stderr.strip() or result.stdout.strip()
    raise RuntimeError(
        f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
    )
