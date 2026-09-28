"""Deployment installer behavior, independent of robot configuration values."""

import importlib.util
from pathlib import Path
import sys
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
    monkeypatch.setattr(
        module, "os", SimpleNamespace(geteuid=lambda: 1234, getuid=lambda: 1234)
    )
    return module, calls


@pytest.fixture
def robot_checkout(monkeypatch, tmp_path):
    checkout = tmp_path / "robot checkout % $"
    for relative in (
        "flake.nix",
        "deps/bello-stack/scripts/setup-can.sh",
        "deps/bello-stack/components/bello-hardwared/config/bello-mini.yaml",
        "deps/bello-stack/components/bello-state-estimatord/config/bello-mini.yaml",
        ".venv/bin/bello-run-reference",
        ".venv/bin/bello-run-hands",
        "runtime/bello-hardwared",
        "runtime/bello-state-estimatord",
        "neutral.npz",
        "nix",
    ):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        "sys.argv",
        [
            "install_systemd.py",
            "robot",
            "--checkout",
            str(checkout),
            "--runtime-bin",
            str(checkout / "runtime"),
            "--nix",
            str(checkout / "nix"),
            "--neutral-model",
            str(checkout / "neutral.npz"),
            "--tracking-endpoint",
            "tracking.example:50051",
            "--human-height",
            "1.7",
            "--confirm-hand-calibration",
        ],
    )
    return checkout


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
    assert not any("sudo" in call for call in calls)
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


def test_robot_installs_can_before_enabling_user_target(
    installer, robot_checkout, monkeypatch
):
    module, calls = installer
    installed = {}

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["sudo", "install"]:
            installed[args[-1]] = Path(args[-2]).read_text()
            assert args[2:9] == ["-D", "-o", "root", "-g", "root", "-m", "644"]

    monkeypatch.setattr(module.subprocess, "run", run)
    module.main()
    assert set(installed) == {
        "/etc/modules-load.d/bello-can.conf",
        "/etc/systemd/system/onboard-can.service",
        "/etc/systemd/system/user@1234.service.d/50-bello-can.conf",
    }
    unit = installed["/etc/systemd/system/onboard-can.service"]
    assert (
        f'RequiresMountsFor="{robot_checkout.as_posix().replace("%", "%%")}"\n' in unit
    )
    script = (robot_checkout / "deps/bello-stack/scripts/setup-can.sh").as_posix()
    escaped_script = script.replace("%", "%%").replace("$", "$$")
    assert f'ExecStart=/usr/bin/bash "{escaped_script}"\n' in unit
    assert (
        "Requires=sys-subsystem-net-devices-can0.device sys-subsystem-net-devices-can1.device\n"
        in unit
    )
    ordering = installed["/etc/systemd/system/user@1234.service.d/50-bello-can.conf"]
    assert "Wants=onboard-can.service\nAfter=onboard-can.service\n" in ordering
    assert calls[0] == ["sudo", "-v"]
    verify = next(
        i for i, call in enumerate(calls) if call[:2] == ["systemd-analyze", "verify"]
    )
    install = next(i for i, call in enumerate(calls) if call[:2] == ["sudo", "install"])
    can_enable = calls.index(["sudo", "systemctl", "enable", "onboard-can.service"])
    user_enable = calls.index(["systemctl", "--user", "enable", "bello-robot.target"])
    assert verify < install < can_enable < user_enable
    assert not any(
        action in call
        for call in calls
        for action in ("start", "stop", "restart", "--now", "devmem", "ip")
    )


def test_robot_dry_run_previews_system_files_without_mutations(
    installer, robot_checkout, monkeypatch, tmp_path, capsys
):
    module, calls = installer
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    module.main()
    preview = capsys.readouterr().out
    assert "/etc/systemd/system/onboard-can.service" in preview
    assert "/etc/modules-load.d/bello-can.conf" in preview
    assert "/etc/systemd/system/user@1234.service.d/50-bello-can.conf" in preview
    assert not (tmp_path / ".config").exists()
    assert calls == []


def test_robot_missing_prerequisite_does_not_mutate(
    installer, robot_checkout, tmp_path
):
    module, calls = installer
    (robot_checkout / ".venv/bin/bello-run-hands").unlink()
    with pytest.raises(ValueError, match="not a file"):
        module.main()
    assert calls == []
    assert not (tmp_path / ".config").exists()


def test_robot_sudo_failure_does_not_write_user_units(
    installer, robot_checkout, monkeypatch, tmp_path
):
    module, calls = installer

    def fail(args, **kwargs):
        calls.append(args)
        raise RuntimeError("sudo authentication failed")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="sudo authentication"):
        module.main()
    assert calls == [["sudo", "-v"]]
    assert not (tmp_path / ".config").exists()


def test_system_verification_failure_prevents_privileged_writes(
    installer, robot_checkout, monkeypatch
):
    module, calls = installer

    def fail(args, **kwargs):
        calls.append(args)
        raise RuntimeError("invalid system unit")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="invalid system unit"):
        module.install_system_files(module.robot_system_files(robot_checkout, 1234))
    assert len(calls) == 1
    assert calls[0][:2] == ["systemd-analyze", "verify"]


def test_root_invocation_is_rejected(installer, robot_checkout, monkeypatch, tmp_path):
    module, calls = installer
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert calls == []
    assert not (tmp_path / ".config").exists()
