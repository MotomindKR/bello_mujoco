import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath

import pytest

from bello_mujoco.training.vast import GitSelection, VastTrainConfig, validate_config
from bello_mujoco.training.vast_remote import (
    remote_start_command,
    remote_upload_command,
    run_remote_training,
    SshExecutor,
)


def config(path, experiment="motion"):
    return VastTrainConfig(
        experiment=experiment, motion_path=path, branch="main", max_price=0.6
    )


def selection():
    return GitSelection(
        "main",
        "a" * 40,
        PurePosixPath("configs/profiles/train/motion.yaml"),
        "https://example.com/robot.git",
    )


@pytest.fixture
def job(tmp_path):
    (tmp_path / "armed").write_text("100")
    (tmp_path / "supervisor_pid").write_text(str(os.getpid()))
    return tmp_path / "run with spaces"


def upload(job, content, digest=None):
    command = remote_upload_command(
        str(job), digest or hashlib.sha256(content).hexdigest()
    )
    return subprocess.run(shlex.split(command), input=content, capture_output=True)


def test_upload_streams_binary_to_atomic_destination_and_retries(
    job, tmp_path, monkeypatch
):
    source = tmp_path / "local untracked.npz"
    content = bytes(range(256)) * 8192
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        "bello_mujoco.training.vast_remote.ssh_command",
        lambda _, command: shlex.split(command),
    )
    result = SshExecutor().upload(
        ("root", "unused", 22), source, str(job), digest, timeout_seconds=10
    )
    assert result.returncode == 0
    assert (job / "motion.npz").read_bytes() == content
    assert upload(job, content).returncode == 0
    assert sorted(p.name for p in job.iterdir()) == ["motion.npz"]


def test_mismatch_preserves_previous_copy_and_never_starts_job(job):
    assert upload(job, b"original").returncode == 0
    result = upload(job, b"truncated", hashlib.sha256(b"complete").hexdigest())
    assert result.returncode != 0
    assert b"SHA-256 mismatch" in result.stderr
    assert (job / "motion.npz").read_bytes() == b"original"
    assert not (job / "pid").exists()
    assert not list(job.glob(".upload-*"))


def test_started_job_allows_identical_retry_but_not_replacement(job):
    assert upload(job, b"original").returncode == 0
    (job / "pid").write_text(str(os.getpid()))
    assert upload(job, b"original").returncode == 0
    assert upload(job, b"different").returncode != 0
    assert (job / "motion.npz").read_bytes() == b"original"


def test_start_rechecks_uploaded_digest_and_passes_remote_not_local_path(job):
    assert upload(job, b"clip").returncode == 0
    command = remote_start_command(
        str(job), selection(), config(Path("/local/clip.npz")), "run"
    )
    args = shlex.split(command)
    assert args[-5] == str(job / "motion.npz")
    assert "/local/clip.npz" not in command
    payload = json.dumps(
        dict(
            environment={},
            script="exit 0",
            timeout_seconds=5,
            shutdown_grace_seconds=0,
            motion_sha256="0" * 64,
        )
    )
    result = subprocess.run(args, input=payload, text=True, capture_output=True)
    assert result.returncode != 0
    assert "uploaded motion changed" in result.stderr
    assert not (job / "pid").exists()


def test_runner_retries_upload_before_start_with_verified_checksum(tmp_path):
    clip = tmp_path / "motion.npz"
    clip.write_bytes(b"local clip")
    events = []

    class Executor:
        attempts = 0

        def upload(self, target, path, job_dir, digest, *, timeout_seconds):
            events.append("upload")
            assert path == clip
            assert digest == hashlib.sha256(clip.read_bytes()).hexdigest()
            self.attempts += 1
            return subprocess.CompletedProcess(
                [], 255 if self.attempts == 1 else 0, stdout="", stderr=""
            )

        def run(self, target, command, *, timeout_seconds, input_text=None):
            if input_text:
                events.append("start")
                assert (
                    json.loads(input_text)["motion_sha256"]
                    == hashlib.sha256(clip.read_bytes()).hexdigest()
                )
                return subprocess.CompletedProcess([], 0, stdout="", stderr="")
            events.append("monitor")
            return subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    dict(state="complete", offset=0, output="", returncode=0)
                ),
                stderr="",
            )

    run_remote_training(
        ("root", "unused", 22),
        selection(),
        config(clip),
        "run",
        wandb_key="secret",
        github_token="",
        executor=Executor(),
        sleep=lambda _: None,
    )
    assert events == ["upload", "upload", "start", "monitor"]


@pytest.mark.parametrize("experiment", ["motion", "bello_mini_motion"])
def test_uploaded_motion_reaches_detached_job_outside_checkout(job, experiment):
    content = b"uploaded reference"
    assert upload(job, content).returncode == 0
    command = remote_start_command(
        str(job), selection(), config(Path("/local/clip.npz"), experiment), "run"
    )
    payload = json.dumps(
        dict(
            environment={},
            script='cat "$6"',
            timeout_seconds=5,
            shutdown_grace_seconds=0,
            motion_sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    subprocess.run(
        shlex.split(command), input=payload, text=True, capture_output=True, check=True
    )
    for _ in range(100):
        if (job.parent / "status").exists():
            break
        time.sleep(0.01)
    assert (job / "status").read_text().strip() == "0"
    assert (job / "output.log").read_bytes() == content


def test_local_paths_are_not_required_to_belong_to_repository(tmp_path, monkeypatch):
    clip = tmp_path / "ignored clip.npz"
    clip.write_bytes(b"clip")
    validate_config(config(clip))
    monkeypatch.chdir(tmp_path)
    validate_config(config(Path(clip.name)))


@pytest.mark.parametrize("name", ["missing.npz", "wrong.txt", "directory.npz"])
def test_bad_local_paths_fail_before_rental(tmp_path, name):
    path = tmp_path / name
    if name == "directory.npz":
        path.mkdir()
    elif name == "wrong.txt":
        path.write_bytes(b"clip")
    with pytest.raises(ValueError, match="existing local .npz"):
        validate_config(config(path))
