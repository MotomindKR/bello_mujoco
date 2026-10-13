#!/usr/bin/env python3
"""Install boot-enabled daemon targets; never start or restart motor IO now."""

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen


HEADER = "# Managed by bello_mujoco/scripts/install_systemd.py\n"
TRACKING_SERVICES = ("xrobotoolkit", "pico-smplx")
ROBOT_SERVICES = (
    "bello-hardwared",
    "bello-state-estimatord",
    "bello-reference",
    "bello-hands",
    "bello-policy",
)

XROBOT_RELEASE_URL = (
    "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0"
)


@dataclass(frozen=True)
class XRobotRelease:
    filename: str
    sha256: str
    sdk_arch: str


def xrobot_release() -> XRobotRelease:
    distro = platform.freedesktop_os_release()
    version = distro.get("VERSION_ID")
    if distro.get("ID") != "ubuntu" or version not in ("22.04", "24.04"):
        raise ValueError("XRoboToolkit packages support Ubuntu 22.04 and 24.04")
    machine = platform.machine()
    if machine == "aarch64":
        return XRobotRelease(
            "XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb",
            "532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f",
            "arm64",
        )
    if machine == "x86_64":
        digest = {
            "22.04": "61961067eb4b41f81ed7cae35f4690dbb0ddfefb329a12b24e0b90ebc46ada91",
            "24.04": "bce661f0be0b8a246ceecb2e5f1675a81c26b834648dc7fdf23f8c0bfe2a5d19",
        }[version]
        return XRobotRelease(
            f"XRoboToolkit_PC_Service_1.0.0_ubuntu_{version}_amd64.deb",
            digest,
            "x64",
        )
    raise ValueError(f"unsupported XRoboToolkit architecture: {machine}")


@dataclass(frozen=True)
class TrackingSetup:
    checkout: Path
    directory: Path
    uv: Path
    release: XRobotRelease
    listen: tuple[str, ...]
    device_id: str | None = None

    @property
    def bridge(self) -> Path:
        return self.checkout / "deps/pico-smplx-bridge"

    @property
    def package(self) -> Path:
        return self.directory / "xrobotoolkit" / self.release.sha256

    @property
    def app(self) -> Path:
        return self.package / "opt/apps/roboticsservice"

    @property
    def sdk(self) -> Path:
        return self.app / "SDK" / self.release.sdk_arch / "libPXREARobotSDK.so"

    @property
    def requirements(self) -> Path:
        return require_file(self.bridge / "requirements.lock")

    @property
    def venv(self) -> Path:
        digest = hashlib.sha256(self.requirements.read_bytes()).hexdigest()
        return self.directory / "venvs" / f"py312-{digest}"

    def files(self, target: str) -> dict[str, str]:
        environment = {
            "LD_LIBRARY_PATH": ":".join(
                str(path) for path in (self.app, self.app / "lib", self.sdk.parent)
            ),
            "QT_PLUGIN_PATH": str(self.app / "plugins"),
        }
        command: list[str | Path] = [
            self.venv / "bin/python",
            require_file(self.bridge / "server.py"),
            "--library",
            self.sdk,
        ]
        for address in self.listen:
            host, separator, port = address.rpartition(":")
            if (
                not host
                or not separator
                or not port.isdigit()
                or not 1 <= int(port) <= 65535
            ):
                raise ValueError(f"invalid bridge listener: {address}")
            command.extend(("--listen", address))
        if self.device_id:
            command.extend(("--device-id", self.device_id))
        return {
            "xrobotoolkit.service": service(
                target,
                "XRoboToolkit headless tracking receiver (no motor IO)",
                self.app,
                [self.app / "RoboticsServiceProcess"],
                environment=environment,
            ),
            "pico-smplx.service": service(
                target,
                "PICO direct body and hand pose bridge (no motor IO)",
                self.bridge,
                command,
                environment=environment,
                after=("xrobotoolkit.service",),
            ),
        }

    def provision(self) -> None:
        # Extract the verified distribution without running vendor desktop installers.
        # Versioned paths keep repeat installs from overwriting running binaries.
        marker = self.package / ".bello-sha256"
        if not marker.exists():
            self.package.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(
                dir=self.package.parent, prefix="download-"
            ) as temporary:
                staging = Path(temporary)
                archive = staging / self.release.filename
                digest = hashlib.sha256()
                with urlopen(
                    f"{XROBOT_RELEASE_URL}/{self.release.filename}", timeout=60
                ) as source:
                    with archive.open("wb") as destination:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            destination.write(chunk)
                if digest.hexdigest() != self.release.sha256:
                    raise ValueError("XRoboToolkit download SHA256 mismatch")
                unpacked = staging / "unpacked"
                subprocess.run(
                    ["dpkg-deb", "--extract", str(archive), str(unpacked)], check=True
                )
                app = unpacked / "opt/apps/roboticsservice"
                require_file(app / "RoboticsServiceProcess")
                require_file(
                    app / "SDK" / self.release.sdk_arch / "libPXREARobotSDK.so"
                )
                (unpacked / marker.name).write_text(self.release.sha256 + "\n")
                unpacked.rename(self.package)
        if marker.read_text().strip() != self.release.sha256:
            raise ValueError(
                f"XRoboToolkit installation checksum marker mismatch: {marker}"
            )
        require_file(self.app / "RoboticsServiceProcess")
        require_file(self.sdk)
        ready = self.venv / ".bello-requirements"
        if not ready.exists():
            subprocess.run(
                [
                    str(self.uv),
                    "venv",
                    "--clear",
                    "--python",
                    "3.12",
                    "--managed-python",
                    str(self.venv),
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(self.uv),
                    "pip",
                    "sync",
                    "--python",
                    str(self.venv / "bin/python"),
                    "--require-hashes",
                    str(self.requirements),
                ],
                check=True,
            )
            ready.write_bytes(self.requirements.read_bytes())
        if ready.read_bytes() != self.requirements.read_bytes():
            raise ValueError(f"bridge environment requirements mismatch: {ready}")
        require_file(self.venv / "bin/python")


def quote(value: str | Path, *, executable: bool = False) -> str:
    text = str(value)
    if any(c in text for c in "\n\r\0"):
        raise ValueError("unit values cannot contain newlines or NUL")
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if executable:
        text = text.replace("$", "$$")
    return f'"{text}"'


def require_file(path: Path) -> Path:
    path = path.absolute()
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return path


def robot_system_files(checkout: Path, uid: int) -> dict[Path, str]:
    deployment = Path(__file__).resolve().parents[1] / "deploy/systemd"
    template = require_file(deployment / "onboard-can.service.in").read_text()
    can_setup = require_file(checkout / "deps/bello-stack/scripts/setup-can.sh")
    return {
        Path("/etc/modules-load.d/bello-can.conf"): HEADER
        + require_file(deployment / "bello-can-modules.conf").read_text(),
        Path("/etc/systemd/system/onboard-can.service"): HEADER
        + template.format(
            checkout=quote(checkout), can_setup=quote(can_setup, executable=True)
        ),
        Path(f"/etc/systemd/system/user@{uid}.service.d/50-bello-can.conf"): HEADER
        + require_file(deployment / "user-can-ordering.conf").read_text(),
    }


def install_system_files(files: dict[Path, str]) -> None:
    with TemporaryDirectory(prefix="bello-systemd-") as temporary:
        staged = Path(temporary)
        for destination, content in files.items():
            (staged / destination.name).write_text(content)
        subprocess.run(
            ["systemd-analyze", "verify", str(staged / "onboard-can.service")],
            check=True,
        )
        for destination in files:
            subprocess.run(
                [
                    "sudo",
                    "install",
                    "-D",
                    "-o",
                    "root",
                    "-g",
                    "root",
                    "-m",
                    "644",
                    str(staged / destination.name),
                    str(destination),
                ],
                check=True,
            )
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    subprocess.run(["sudo", "systemctl", "enable", "onboard-can.service"], check=True)


def service(
    target: str,
    description: str,
    checkout: Path,
    command: list[str | Path],
    *,
    restart: bool = True,
    preflight: list[str | Path] | None = None,
    environment: dict[str, str] | None = None,
    after: tuple[str, ...] = (),
) -> str:
    before_start = (
        "ExecStartPre="
        + " ".join(quote(arg, executable=True) for arg in preflight)
        + "\n"
        if preflight
        else ""
    )
    ordering = (
        "After=" + " ".join(after) + "\nWants=" + " ".join(after) + "\n"
        if after
        else ""
    )
    environment = environment or {}
    environment_lines = "".join(
        f"Environment={quote(f'{key}={value}')}\n" for key, value in environment.items()
    )
    if "LD_LIBRARY_PATH" not in environment:
        environment_lines += "UnsetEnvironment=LD_LIBRARY_PATH\n"
    return (
        HEADER
        + f"""[Unit]
Description={description}
PartOf={target}
{ordering}StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=exec
WorkingDirectory={quote(checkout)[1:-1]}
{before_start}ExecStart={" ".join(quote(arg, executable=True) for arg in command)}
Environment=PYTHONUNBUFFERED=1
Environment=JAX_PLATFORMS=cpu
{environment_lines}Restart={"on-failure" if restart else "no"}
RestartSec=3
KillSignal=SIGINT
KillMode=control-group
TimeoutStopSec=20
StandardOutput=journal
StandardError=journal
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    roles = parser.add_subparsers(dest="role", required=True)
    robot = roles.add_parser("robot", help="install robot daemons and Jetson CAN setup")
    tracking_parser = roles.add_parser(
        "tracking", help="install native tracking receiver and bridge"
    )
    for role in (robot, tracking_parser):
        role.add_argument("--checkout", type=Path, required=True)
        role.add_argument("--dry-run", action="store_true")
    robot.add_argument(
        "--runtime-bin",
        type=Path,
        default=Path.home() / ".cache/bello-mujoco/rust-runtime/release",
    )
    robot.add_argument(
        "--nix", type=Path, default=Path("/nix/var/nix/profiles/default/bin/nix")
    )
    robot.add_argument("--tracking-endpoint", required=True)
    robot.add_argument("--human-height", type=float, required=True)
    robot.add_argument("--hardware-config", type=Path)
    robot.add_argument("--estimator-config", type=Path)
    robot.add_argument(
        "--policy", type=Path, required=True, help="locomotion bundle ZIP"
    )
    robot.add_argument(
        "--policy-source",
        type=Path,
        required=True,
        help="exact clean bundle source checkout",
    )
    robot.add_argument(
        "--policy-config", type=Path, required=True, help="runner hardware/control YAML"
    )
    robot.add_argument("--confirm-hand-calibration", action="store_true", required=True)
    tracking_parser.add_argument(
        "--uv", type=Path, help="uv executable for setup (default: PATH)"
    )
    tracking_parser.add_argument(
        "--tracking-dir", type=Path, default=Path.home() / ".local/share/bello/tracking"
    )
    tracking_parser.add_argument(
        "--bridge-listen",
        action="append",
        help="repeatable listener (default: 0.0.0.0:50052)",
    )
    tracking_parser.add_argument(
        "--device-id", help="track only this headset device ID"
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("install on the Linux service host, as the operator account")
    if os.geteuid() == 0:
        parser.error(
            "run as the operator account; robot installation uses sudo for CAN setup"
        )

    target = f"bello-{args.role}.target"
    system_files = {}
    tracking = None
    services = ROBOT_SERVICES if args.role == "robot" else TRACKING_SERVICES
    files = {
        target: HEADER
        + "[Unit]\n"
        + f"Description=Bello {args.role} daemons\n"
        + "Wants="
        + " ".join(f"{name}.service" for name in services)
        + "\n"
        + "After="
        + " ".join(f"{name}.service" for name in services)
        + "\n\n[Install]\nWantedBy=default.target\n"
    }
    if args.role == "tracking":
        uv = args.uv or shutil.which("uv")
        if not uv:
            parser.error(
                "tracking setup requires uv; run inside nix develop or pass --uv"
            )
        tracking = TrackingSetup(
            args.checkout.resolve(strict=True),
            args.tracking_dir.absolute(),
            require_file(Path(uv)),
            xrobot_release(),
            tuple(args.bridge_listen or ("0.0.0.0:50052",)),
            args.device_id,
        )
        # Resolve all source prerequisites before downloading or writing anything.
        require_file(tracking.requirements)
        files.update(tracking.files(target))
    else:
        if not 0.5 <= args.human_height <= 2.5:
            parser.error("human height must be in meters, between 0.5 and 2.5")
        checkout = args.checkout.resolve(strict=True)
        require_file(checkout / "flake.nix")
        nix = require_file(args.nix)
        can_check = [
            "/usr/bin/bash",
            require_file(checkout / "deps/bello-stack/scripts/setup-can.sh"),
            "--check",
        ]
        for name, override in (
            ("bello-hardwared", args.hardware_config),
            ("bello-state-estimatord", args.estimator_config),
        ):
            config = require_file(
                override
                or checkout
                / f"deps/bello-stack/components/{name}/config/bello-mini.yaml"
            )
            binary = require_file(args.runtime_bin / name)
            files[f"{name}.service"] = service(
                target,
                name,
                checkout,
                [
                    nix,
                    "develop",
                    ".#rust-runtime",
                    "--command",
                    binary,
                    "--config",
                    config,
                ],
                preflight=can_check if name == "bello-hardwared" else None,
            )
        files["bello-reference.service"] = service(
            target,
            "Bello GMR body reference producer (no motor IO)",
            checkout,
            [
                require_file(checkout / ".venv/bin/bello-run-reference"),
                "asset=bello_mini",
                f"xrobot.endpoint={args.tracking_endpoint}",
                f"retarget.human_height_m={args.human_height}",
            ],
        )
        files["bello-hands.service"] = service(
            target,
            "Bello calibrated Linker O6 hands (ACTIVE CAN CONTROL)",
            checkout,
            [
                require_file(checkout / ".venv/bin/bello-run-hands"),
                f"xrobot.endpoint={args.tracking_endpoint}",
                "protocol=linker_o6",
                "left_can_interface=can2",
                "right_can_interface=can3",
                "arm=true",
                "calibration_confirmed=true",
            ],
            restart=False,
        )
        policy_source = args.policy_source.resolve(strict=True)
        if not policy_source.is_dir():
            raise ValueError(f"not a source directory: {policy_source}")
        files["bello-policy.service"] = (
            service(
                target,
                "Bello policy operator service (gamepad Hold chord required)",
                checkout,
                [
                    nix,
                    "develop",
                    ".#rust-runtime",
                    "--command",
                    "bash",
                    require_file(checkout / "scripts/run_policy_service.sh"),
                    require_file(args.runtime_bin / "bello-policy"),
                    require_file(args.policy),
                    policy_source,
                    require_file(args.policy_config),
                ],
                after=tuple(
                    f"{name}.service"
                    for name in ROBOT_SERVICES
                    if name != "bello-policy"
                ),
            )
            + "LogsDirectory=bello-policy\nLogsDirectoryMode=0700\n"
        )
        system_files = robot_system_files(checkout, os.getuid())

    unit_dir = Path.home() / ".config/systemd/user"
    for name, content in files.items():
        path = unit_dir / name
        if path.exists() and not path.read_text().startswith(HEADER):
            raise RuntimeError(f"refusing to overwrite unmanaged file: {path}")
        if args.dry_run:
            print(f"--- {path}\n{content}")
    if args.dry_run:
        if tracking:
            print(f"Download: {XROBOT_RELEASE_URL}/{tracking.release.filename}")
            print(f"SHA256: {tracking.release.sha256}")
            print(f"Extract package (no vendor install scripts): {tracking.package}")
            print(f"Prepare Python 3.12 environment: {tracking.venv}")
            print(f"Requirements: {tracking.requirements}")
        for path, content in system_files.items():
            print(f"--- {path}\n{content}")
        return
    if system_files:
        subprocess.run(["sudo", "-v"], check=True)
    if tracking:
        tracking.provision()
    for name, content in files.items():
        path = unit_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(
        [
            "systemd-analyze",
            "--user",
            "verify",
            str(unit_dir / target),
            *(f"{name}.service" for name in services),
        ],
        check=True,
    )
    if system_files:
        install_system_files(system_files)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    # Only the target is boot-enabled. Neither command changes running processes.
    subprocess.run(
        [
            "systemctl",
            "--user",
            "disable",
            *(f"{name}.service" for name in services),
        ],
        check=True,
    )
    subprocess.run(["systemctl", "--user", "enable", target], check=True)
    print(
        f"Installed and boot-enabled {target}; no process was started, stopped, or restarted."
    )
    if system_files:
        print(
            "Installed and boot-enabled onboard-can.service for can0-can4 initialization."
        )
    print("User lingering must be enabled to start at boot without a login.")
    print(
        f"After checking robot safety and stopping unmanaged duplicates: systemctl --user start {target}"
    )


if __name__ == "__main__":
    main()
