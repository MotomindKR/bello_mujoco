from pathlib import PurePosixPath
from types import SimpleNamespace
import urllib.error

import pytest

from bello_mujoco.training import vast, vast_host


@pytest.mark.parametrize("exit_code", [0, 1, 137])
def test_host_tears_down_finished_or_failed_jobs(tmp_path, exit_code):
    (tmp_path / "status").write_text(str(exit_code))
    assert (
        vast_host.shutdown_reason(tmp_path, 100, 10)
        == f"job exited with status {exit_code}"
    )


def test_host_deadline_does_not_depend_on_local_monitor(tmp_path):
    assert vast_host.shutdown_reason(tmp_path, 100, 99) is None
    assert vast_host.shutdown_reason(tmp_path, 100, 100) == "hard timeout"


def test_host_detects_job_disappearing_without_exit_status(monkeypatch, tmp_path):
    (tmp_path / "job_pid").write_text("42")

    def missing(pid, signal):
        assert (pid, signal) == (42, 0)
        raise ProcessLookupError

    monkeypatch.setattr(vast_host.os, "kill", missing)
    assert vast_host.shutdown_reason(tmp_path, 100, 10) == "job process disappeared"


def test_host_detects_zombie_job(monkeypatch, tmp_path):
    (tmp_path / "job_pid").write_text("42")
    proc = tmp_path / "stat"
    proc.write_text("42 (bash) Z 1 42 42 0")
    monkeypatch.setattr(vast_host.os, "kill", lambda *args: None)
    monkeypatch.setattr(vast_host, "Path", lambda path: proc)
    assert vast_host.shutdown_reason(tmp_path, 100, 10) == "job process is a zombie"


def test_host_supervisor_retries_self_teardown(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTAINER_ID", "42")
    monkeypatch.setenv("CONTAINER_API_KEY", "instance-scoped-token")
    reasons = iter([None, "job exited with status 0"])
    monkeypatch.setattr(vast_host, "shutdown_reason", lambda *args: next(reasons))
    requests, sleeps = [], []

    def destroy(instance_id, key):
        requests.append((instance_id, key))
        return len(requests) == 2

    monkeypatch.setattr(vast_host, "destroy_once", destroy)
    monkeypatch.setattr(vast_host.time, "sleep", sleeps.append)
    vast_host.supervise(100, tmp_path)
    assert requests == [("42", "instance-scoped-token")] * 2
    assert sleeps == [2, 30, 5]
    assert (tmp_path / "armed").read_text() == "100"


def test_host_requires_its_scoped_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        vast_host.supervise(100, tmp_path)
    assert not (tmp_path / "armed").exists()


def test_host_delete_is_scoped_to_own_instance(monkeypatch):
    requests = []

    def send(request, timeout):
        requests.append((request, timeout))
        raise urllib.error.HTTPError(request.full_url, 404, "gone", None, None)

    monkeypatch.setattr(vast_host.urllib.request, "urlopen", send)
    assert vast_host.destroy_once("42", "token")
    request, timeout = requests[0]
    assert request.full_url.endswith("/instances/42/")
    assert request.get_method() == "DELETE"
    assert request.get_header("Authorization") == "Bearer token"
    assert timeout == 30


@pytest.mark.parametrize("error", [None, KeyboardInterrupt(), RuntimeError("SSH lost")])
def test_local_monitor_exit_never_destroys_remote_run(monkeypatch, tmp_path, error):
    api = SimpleNamespace(destroy_instance=lambda *args: pytest.fail("local teardown"))
    selection = vast.GitSelection(
        "main", "a" * 40, PurePosixPath("config"), "https://example.com/repo.git"
    )
    monkeypatch.setattr(vast, "resolve_git_selection", lambda *args: selection)
    monkeypatch.setattr(vast, "credential", lambda *args, **kwargs: "token")
    monkeypatch.setattr(vast, "find_offers", lambda *args: [])
    monkeypatch.setattr(vast, "create_available_instance", lambda *args: 42)
    monkeypatch.setattr(vast, "wait_for_instance", lambda *args: None)
    monkeypatch.setattr(vast, "wait_for_ssh", lambda *args: ("root", "host", 22))

    def monitor(*args, **kwargs):
        if error is not None:
            raise error

    config = vast.VastTrainConfig(max_price=0.5, branch="main")
    if error is None:
        assert (
            vast.launch_pipeline(
                tmp_path, config, api_factory=lambda key: api, remote_runner=monitor
            )
            == 0
        )
    else:
        with pytest.raises(type(error)):
            vast.launch_pipeline(
                tmp_path, config, api_factory=lambda key: api, remote_runner=monitor
            )
