#!/usr/bin/env python3
"""Install boot-enabled daemon targets; never start or restart motor IO now."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


HEADER = "# Managed by bello_mujoco/scripts/install_systemd.py\n"
TRACKING_SERVICES = ("xrobotoolkit", "pico-smplx", "pico-smplx-viewer")
ROBOT_SERVICES = (
    "bello-hardwared",
    "bello-state-estimatord",
    "bello-reference",
    "bello-hands",
)


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
) -> str:
    before_start = (
        "ExecStartPre="
        + " ".join(quote(arg, executable=True) for arg in preflight)
        + "\n"
        if preflight
        else ""
    )
    return (
        HEADER
        + f"""[Unit]
Description={description}
PartOf={target}
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=exec
WorkingDirectory={quote(checkout)[1:-1]}
{before_start}ExecStart={" ".join(quote(arg, executable=True) for arg in command)}
Environment=PYTHONUNBUFFERED=1
Environment=JAX_PLATFORMS=cpu
UnsetEnvironment=LD_LIBRARY_PATH
Restart={"on-failure" if restart else "no"}
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
    parser.add_argument("role", choices=("robot", "tracking"))
    parser.add_argument("--checkout", type=Path)
    parser.add_argument(
        "--runtime-bin",
        type=Path,
        default=Path.home() / ".cache/bello-mujoco/rust-runtime/release",
    )
    parser.add_argument(
        "--nix", type=Path, default=Path("/nix/var/nix/profiles/default/bin/nix")
    )
    parser.add_argument("--tracking-endpoint")
    parser.add_argument("--neutral-model", type=Path)
    parser.add_argument("--human-height", type=float)
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--estimator-config", type=Path)
    parser.add_argument("--confirm-hand-calibration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("install on the Linux service host, as the operator account")
    if os.geteuid() == 0:
        parser.error(
            "run as the operator account; robot installation uses sudo for CAN setup"
        )

    target = f"bello-{args.role}.target"
    system_files = {}
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
        for name in services:
            subprocess.run(
                ["systemctl", "--user", "cat", f"{name}.service"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            files[f"{name}.service.d/50-bello-lifecycle.conf"] = (
                HEADER + f"[Unit]\nPartOf={target}\n" + "\n[Install]\nWantedBy=\n"
            )
    else:
        if not (
            args.checkout
            and args.tracking_endpoint
            and args.neutral_model
            and args.human_height
            and args.confirm_hand_calibration
        ):
            parser.error(
                "robot requires --checkout, --tracking-endpoint, --neutral-model, --human-height and --confirm-hand-calibration"
            )
        if not 0.5 <= args.human_height <= 2.5:
            parser.error("human height must be in meters, between 0.5 and 2.5")
        checkout = args.checkout.resolve(strict=True)
        require_file(checkout / "flake.nix")
        neutral_model = require_file(args.neutral_model)
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
                f"smplx.neutral_model_path={neutral_model}",
                f"smplx.endpoint={args.tracking_endpoint}",
                "protocol=linker_o6",
                "can_interface=can1",
                "arm=true",
                "calibration_confirmed=true",
            ],
            restart=False,
            preflight=can_check,
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
        for path, content in system_files.items():
            print(f"--- {path}\n{content}")
        return
    if system_files:
        subprocess.run(["sudo", "-v"], check=True)
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
            "Installed and boot-enabled onboard-can.service for can0/can1 initialization."
        )
    print("User lingering must be enabled to start at boot without a login.")
    print(
        f"After checking robot safety and stopping unmanaged duplicates: systemctl --user start {target}"
    )


if __name__ == "__main__":
    main()
