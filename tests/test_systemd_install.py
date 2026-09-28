"""Deployment installer behavior, independent of robot configuration values."""

import importlib.util
import hashlib
import io
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
    monkeypatch.setitem(sys.modules, spec.name, module)
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
            "--tracking-endpoint",
            "tracking.example:50051",
            "--human-height",
            "1.7",
            "--confirm-hand-calibration",
        ],
    )
    return checkout


def test_tracking_install_never_starts_or_stops_processes(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    monkeypatch.setattr(module.TrackingSetup, "provision", lambda self: None)
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


def test_dry_run_does_not_write_or_reload(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    module.main()
    assert not (tmp_path / ".config").exists()
    assert calls == []


def test_install_refuses_unmanaged_overwrite(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    target = tmp_path / ".config/systemd/user/bello-tracking.target"
    target.parent.mkdir(parents=True)
    target.write_text("user-owned unit\n")
    with pytest.raises(RuntimeError, match="unmanaged"):
        module.main()
    assert target.read_text() == "user-owned unit\n"
    assert not list(target.parent.glob("*.service.d"))
    assert calls == []


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


@pytest.fixture
def tracking_checkout(installer, monkeypatch, tmp_path):
    module, calls = installer
    checkout = tmp_path / "checkout"
    bridge = checkout / "deps/pico-smplx-bridge"
    bridge.mkdir(parents=True)
    (bridge / "server.py").touch()
    (bridge / "requirements.lock").write_text("grpcio==1.84.0\n")
    uv = tmp_path / "uv"
    uv.touch()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        module.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "22.04"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install_systemd.py",
            "tracking",
            "--checkout",
            str(checkout),
            "--uv",
            str(uv),
        ],
    )
    return checkout


def test_tracking_units_use_direct_sdk_and_separate_local_port(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    monkeypatch.setattr(module.TrackingSetup, "provision", lambda self: None)
    module.main()
    units = tmp_path / ".config/systemd/user"
    receiver = (units / "xrobotoolkit.service").read_text()
    bridge = (units / "pico-smplx.service").read_text()
    target = (units / "bello-tracking.target").read_text()
    assert "viewer" not in target
    assert "RoboticsServiceProcess" in receiver
    assert "SDK/arm64/libPXREARobotSDK.so" in bridge
    assert '"--listen" "127.0.0.1:50052"' in bridge
    assert "After=xrobotoolkit.service" in bridge
    assert "Wants=xrobotoolkit.service" in bridge
    assert "UnsetEnvironment=LD_LIBRARY_PATH" not in receiver + bridge
    assert "calibration" not in bridge
    assert "uv " not in receiver + bridge
    assert "nix " not in receiver + bridge


@pytest.mark.parametrize(
    "machine,version,sdk",
    [
        ("aarch64", "22.04", "arm64"),
        ("x86_64", "22.04", "x64"),
        ("x86_64", "24.04", "x64"),
    ],
)
def test_native_release_selection(installer, monkeypatch, machine, version, sdk):
    module, _ = installer
    monkeypatch.setattr(module.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        module.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": version},
    )
    release = module.xrobot_release()
    assert release.sdk_arch == sdk
    assert len(release.sha256) == 64


def test_tracking_rejects_unknown_platform_before_writing(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    monkeypatch.setattr(module.platform, "machine", lambda: "riscv64")
    with pytest.raises(ValueError, match="architecture"):
        module.main()
    assert calls == []
    assert not (tmp_path / ".config").exists()


def test_tracking_missing_source_prevents_provision(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    (tracking_checkout / "deps/pico-smplx-bridge/server.py").unlink()
    with pytest.raises(ValueError, match="not a file"):
        module.main()
    assert calls == []
    assert not (tmp_path / ".config").exists()


def test_tracking_download_checksum_failure_never_extracts_or_installs(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    setup = module.TrackingSetup(
        tracking_checkout,
        tmp_path / "tracking",
        tmp_path / "uv",
        module.xrobot_release(),
        ("127.0.0.1:50052",),
    )
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupted"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        setup.provision()
    assert calls == []
    assert not setup.package.exists()
    assert not setup.venv.exists()


def test_tracking_repeat_provision_reuses_completed_package_and_environment(
    installer, tracking_checkout, monkeypatch, tmp_path
):
    module, calls = installer
    payload = b"synthetic package"
    release = module.XRobotRelease(
        "synthetic.deb", hashlib.sha256(payload).hexdigest(), "arm64"
    )
    setup = module.TrackingSetup(
        tracking_checkout,
        tmp_path / "tracking",
        tmp_path / "uv",
        release,
        ("127.0.0.1:50052",),
    )
    downloads = []

    def download(url, **kwargs):
        downloads.append(url)
        return io.BytesIO(payload)

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["dpkg-deb", "--extract"]:
            app = Path(args[-1]) / "opt/apps/roboticsservice"
            sdk = app / "SDK/arm64"
            sdk.mkdir(parents=True)
            (sdk / "libPXREARobotSDK.so").touch()
            (app / "RoboticsServiceProcess").touch()
        elif args[1] == "venv":
            (setup.venv / "bin").mkdir(parents=True)
            (setup.venv / "bin/python").touch()

    monkeypatch.setattr(module, "urlopen", download)
    monkeypatch.setattr(module.subprocess, "run", run)
    setup.provision()
    original_calls = list(calls)
    setup.provision()
    assert len(downloads) == 1
    assert calls == original_calls
    assert len(calls) == 3  # extraction, venv creation, pinned dependency sync
