"""Deployment installer behavior, independent of robot configuration values."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installer(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "install_systemd", Path(__file__).parents[1] / "scripts/install_systemd.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(
        module,
        "subprocess",
        SimpleNamespace(run=lambda args, **kw: calls.append(args), DEVNULL=-3),
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    return module, calls


def test_tracking_install_never_starts_or_stops_processes(
    installer, monkeypatch, tmp_path
):
    module, calls = installer
    monkeypatch.setattr("sys.argv", ["install_systemd.py", "tracking"])
    module.main()
    assert not any(
        action in call
        for call in calls
        for action in ("start", "stop", "restart", "--now")
    )
    assert any("verify" in call for call in calls)
    assert any("disable" in call for call in calls)
    assert [call for call in calls if "enable" in call] == [
        ["systemctl", "--user", "enable", "bello-tracking.target"]
    ]
    target = (tmp_path / ".config/systemd/user/bello-tracking.target").read_text()
    assert "[Install]\nWantedBy=default.target\n" in target


def test_dry_run_does_not_write_or_reload(installer, monkeypatch, tmp_path):
    module, calls = installer
    monkeypatch.setattr("sys.argv", ["install_systemd.py", "tracking", "--dry-run"])
    module.main()
    assert not (tmp_path / ".config").exists()
    assert all("cat" in call for call in calls)


def test_install_refuses_unmanaged_overwrite(installer, monkeypatch, tmp_path):
    module, calls = installer
    target = tmp_path / ".config/systemd/user/bello-tracking.target"
    target.parent.mkdir(parents=True)
    target.write_text("user-owned unit\n")
    monkeypatch.setattr("sys.argv", ["install_systemd.py", "tracking"])
    with pytest.raises(RuntimeError, match="unmanaged"):
        module.main()
    assert target.read_text() == "user-owned unit\n"
    assert not list(target.parent.glob("*.service.d"))
    assert all("cat" in call for call in calls)


def test_unit_encoding_and_explicit_restart(installer):
    module, _ = installer
    text = module.service(
        "example.target",
        "example",
        Path("/tmp/source tree"),
        ["/tmp/program", 'a% b$"c'],
        restart=False,
        preflight=["/tmp/check script", "--check"],
    )
    assert "WorkingDirectory=/tmp/source tree\n" in text
    assert '"a%% b$$\\"c"' in text
    assert "Restart=no\n" in text
    assert 'ExecStartPre="/tmp/check script" "--check"\n' in text
    assert "PartOf=example.target\n" in text
    with pytest.raises(ValueError, match="newlines"):
        module.quote("bad\nExecStart=command")


def test_required_file_keeps_stable_symlink_path(installer, tmp_path):
    module, _ = installer
    binary = tmp_path / "versioned-binary"
    binary.touch()
    link = tmp_path / "current-binary"
    link.symlink_to(binary)
    assert module.require_file(link) == link
