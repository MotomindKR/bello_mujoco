"""Simulation-only barbell mode with real hand RPCs and synthetic hand devices."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import grpc

from bello_mujoco.barbell import (
    BarbellMode,
    BarbellModeConfig,
    BarbellPhase,
    HandoffConfirmation,
)
from bello_mujoco.hardware.barbell_hands import BarbellHandSession
from bello_mujoco.hardware.hand_server import (
    HandController,
    HandControlClient,
    HandServerConfig,
    hand_service_handler,
)
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.xrobot_hands import XRobotHandsConfig


@dataclass(frozen=True)
class BarbellEvaluationConfig:
    motion_source: Path
    motion_checkpoint: Path
    standing_source: Path
    standing_policy: Path
    native_binary: Path
    onnxruntime_library: Path
    neutral_pose_path: Path
    output_path: Path
    bar_mass_kg: float = 15.0
    holder_move_seconds: float = 2.0
    receive_wait_seconds: float = 4.0
    support_wait_seconds: float = 3.0
    hand_wait_seconds: float = 0.8
    maximum_seconds: float = 90.0
    worker_timeout_seconds: float = 180.0
    mode: BarbellModeConfig = field(default_factory=BarbellModeConfig)

    def __post_init__(self):
        for name in (
            "bar_mass_kg",
            "holder_move_seconds",
            "receive_wait_seconds",
            "support_wait_seconds",
            "hand_wait_seconds",
            "maximum_seconds",
            "worker_timeout_seconds",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.receive_wait_seconds < self.holder_move_seconds:
            raise ValueError(
                "receive_wait_seconds must allow the holder to finish moving"
            )


class _SimulationClock:
    now = 0.0

    def __call__(self):
        return self.now


class _SimulatedHands:
    """Byte-position actuator surrogate, not a contact or gripping model."""

    config = SimpleNamespace(feedback_timeout_seconds=0.5)

    def __init__(self, clock):
        self.clock = clock
        self.positions = {side: (255,) * 6 for side in ("left", "right")}
        self.feedback_at = {side: clock() for side in self.positions}
        self.writes = []

    def feedback(self, side):
        return self.positions[side], self.feedback_at[side]

    def poll_feedback(self, side):
        self.feedback_at[side] = self.clock()

    def refresh_positions(self, side):
        self.poll_feedback(side)
        return self.positions[side]

    def set_positions(self, side, positions):
        self.positions[side] = tuple(positions)
        self.writes.append(
            {"time_s": self.clock(), "side": side, "positions": positions}
        )


def _source_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def _validate_sources(config):
    for name in (
        "motion_source",
        "motion_checkpoint",
        "standing_source",
        "standing_policy",
        "native_binary",
        "onnxruntime_library",
        "neutral_pose_path",
    ):
        if not getattr(config, name).exists():
            raise ValueError(f"{name} does not exist: {getattr(config, name)}")
    import onnx

    with ZipFile(config.standing_policy) as archive:
        graph = onnx.load_from_string(archive.read("policy.onnx"))
    metadata = [entry.value for entry in graph.metadata_props if entry.key == "bello"]
    if len(metadata) != 1:
        raise ValueError(
            "Standing bundle must contain exactly one Bello metadata document"
        )
    standing = json.loads(metadata[0])
    motion = json.loads((config.motion_checkpoint / "metadata.json").read_text())
    for source, expected in (
        (config.motion_source, motion["git_commit"]),
        (config.standing_source, standing["git_commit"]),
    ):
        if _source_commit(source) != expected:
            raise ValueError(
                f"Policy source commit does not match checkpoint: {source}"
            )
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if dirty.strip():
            raise ValueError(
                f"Policy source checkout has tracked modifications: {source}"
            )
    # This diagnostic reuses observation helpers in the motion worker. Require
    # byte-identical standing helpers rather than silently mixing definitions.
    for relative in (
        "envs/observations.py",
        "envs/feedback.py",
        "envs/actions.py",
        "imu.py",
        "models/target_limits.py",
    ):
        path = Path("src/bello_mujoco") / relative
        if (config.motion_source / path).read_bytes() != (
            config.standing_source / path
        ).read_bytes():
            raise ValueError(f"Historical policy helpers differ: {relative}")
    return {
        "standing_source": standing["git_commit"],
        "motion_source": motion["git_commit"],
    }, standing


def _read_worker(process, timeout):
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError(
            "Barbell simulation worker did not respond; inspect worker.log"
        )
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Barbell simulation worker stopped; inspect worker.log")
    return json.loads(line)


def _confirmation(command, row, hand_status, config):
    event = command.awaiting
    elapsed = command.elapsed_seconds
    at_target = all(
        list(hand.measured_positions) == list(hand.commanded_positions)
        and len(hand.measured_positions) == 6
        and all(
            abs(value - round(255 * (1 - command.finger_closure))) <= 1
            for value in hand.measured_positions
        )
        for hand in hand_status.hands
    )
    if event == HandoffConfirmation.BAR_READY:
        ready = (
            elapsed >= config.receive_wait_seconds
            and row["external_support"]
            and row["hand_bar_gap_m"] < 0.025
        )
    elif event == HandoffConfirmation.GRASP_SECURED:
        ready = (
            elapsed >= config.hand_wait_seconds
            and at_target
            and row["hand_bar_gap_m"] < 0.025
            and row["external_support"]
        )
    elif event == HandoffConfirmation.BAR_SUPPORTED:
        ready = elapsed >= config.support_wait_seconds
    elif event == HandoffConfirmation.HANDS_OPEN:
        ready = (
            elapsed >= config.hand_wait_seconds
            and at_target
            and row["external_support"]
            and command.load_fraction == 0
        )
    elif event == HandoffConfirmation.BAR_CLEAR:
        ready = (
            elapsed >= config.holder_move_seconds
            and row["hand_bar_gap_m"] > 0.3
            and row["external_support"]
        )
    else:
        ready = False
    return event if ready else None


def validate_barbell_trace(
    rows: list[dict], confirmations: list[dict], expected_motion_seconds: float
) -> dict:
    """Audit the saved control trace separately from the phase machine."""
    motion_rows = [row for row in rows if row["phase"] == "motion"]
    carried = [
        row for row in rows if row["phase"] in ("to_motion", "motion", "to_standing")
    ]
    checks = {
        "full_motion": bool(motion_rows)
        and math.isclose(
            motion_rows[-1]["motion_time"], expected_motion_seconds, abs_tol=1e-6
        ),
        "all_confirmations": [event["event"] for event in confirmations]
        == [event.value for event in HandoffConfirmation],
        "no_unintended_contacts": bool(rows)
        and all(row["barbell/unintended_contact_force_n"] <= 1 for row in rows),
        "unsupported_during_motion": bool(carried)
        and all(
            not row["external_support"] and row["load_fraction"] == 1 for row in carried
        ),
        "closed_while_loaded": bool(rows)
        and all(row["finger_closure"] == 1 for row in rows if row["load_fraction"] > 0),
        "supported_before_opening": any(row["phase"] == "open_hands" for row in rows)
        and all(
            row["load_fraction"] == 0 and row["external_support"]
            for row in rows
            if row["phase"] == "open_hands"
        ),
        "finished_unloaded": bool(rows)
        and rows[-1]["phase"] == "normal_standing_final"
        and rows[-1]["load_fraction"] == 0
        and rows[-1]["arm_fraction"] == 0,
    }
    required_gates = {
        "bar_ready": "close_hands",
        "grasp_secured": "transfer_load",
        "bar_supported": "transfer_to_holder",
        "hands_open": "await_bar_clear",
        "bar_clear": "hands_normal",
    }
    checks["confirmations_before_actions"] = checks["all_confirmations"] and all(
        any(row["phase"] == required_gates[event["event"]] for row in rows)
        and event["time_s"]
        <= min(
            row["time_s"]
            for row in rows
            if row["phase"] == required_gates[event["event"]]
        )
        for event in confirmations
    )
    return {"passed": all(checks.values()), **checks}


def run_barbell_evaluation(config: BarbellEvaluationConfig) -> dict:
    """No CAN or external hand endpoint is accepted by this evaluation command."""
    provenance, standing = _validate_sources(config)
    out = config.output_path.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("output_path already contains a barbell evaluation")
    settings = asdict(config)
    for name, value in vars(config).items():
        if isinstance(value, Path):
            settings[name] = str(value.resolve())
    settings["output_path"] = str(out)
    settings["standing_metadata"] = str(out / "standing_metadata.json")
    (out / "standing_metadata.json").write_text(json.dumps(standing, indent=2))
    (out / "manifest.json").write_text(json.dumps(settings, indent=2))
    environment = dict(
        os.environ,
        PYTHONPATH=str(config.motion_source.resolve() / "src"),
        JAX_PLATFORMS="cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        ORT_DYLIB_PATH=str(config.onnxruntime_library.resolve()),
    )
    import mujoco

    environment[
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    ] = str(Path(mujoco.__file__).parent)
    worker = Path(__file__).resolve().parents[3] / "tools/barbell/worker.py"
    clock = _SimulationClock()
    devices = _SimulatedHands(clock)
    hand_config = XRobotHandsConfig(
        xrobot=PicoGrpcConfig(endpoint="127.0.0.1:1"),
        arm=True,
        calibration_confirmed=True,
    )
    controller = HandController(hand_config, HandServerConfig(), devices, clock=clock)
    mode = BarbellMode(config.mode)
    confirmations, events, rows = [], [], []
    failure = None
    process = None
    result = {}
    with (
        (out / "worker.log").open("w") as log,
        ThreadPoolExecutor(max_workers=4) as executor,
    ):
        server = None
        try:
            process = subprocess.Popen(
                [sys.executable, str(worker), str(out / "manifest.json")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=environment,
                cwd=out,
            )
            ready = _read_worker(process, config.worker_timeout_seconds)
            dt = ready["dt"]
            server = grpc.server(executor)
            server.add_generic_rpc_handlers((hand_service_handler(controller),))
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            last_hand_tick = -math.inf
            with (
                grpc.insecure_channel(f"127.0.0.1:{port}") as channel,
                BarbellHandSession(HandControlClient(channel), clock=clock) as hands,
            ):
                while (
                    clock.now < config.maximum_seconds
                    and mode.phase != BarbellPhase.COMPLETE
                ):
                    command = mode.advance(dt)
                    if not events or events[-1]["phase_id"] != command.phase_id:
                        events.append(
                            {
                                "phase": command.phase,
                                "phase_id": command.phase_id,
                                "time_s": clock.now,
                            }
                        )
                        print(f"{clock.now:.2f}s {command.phase}", flush=True)
                    clock.now += dt
                    hands.update(command.finger_closure)
                    if (
                        clock.now - last_hand_tick + 1e-9
                        >= 1 / hand_config.control_rate_hz
                    ):
                        controller.tick()
                        last_hand_tick = clock.now
                    hand_status = hands.update(command.finger_closure)
                    process.stdin.write(json.dumps(asdict(command)) + "\n")
                    process.stdin.flush()
                    row = _read_worker(process, config.worker_timeout_seconds)
                    rows.append(row)
                    if row["failure"]:
                        failure = row["failure"]
                        mode.fault(failure)
                        break
                    event = _confirmation(command, row, hand_status, config)
                    if event is not None:
                        mode.confirm(event, phase_id=command.phase_id)
                        confirmations.append(
                            {
                                "event": event,
                                "phase_id": command.phase_id,
                                "time_s": clock.now,
                                "source": "simulation_holder_and_ideal_grasp",
                                "hand_bar_gap_m": row["hand_bar_gap_m"],
                            }
                        )
                if mode.phase != BarbellPhase.COMPLETE and failure is None:
                    failure = "confirmation_or_cycle_timeout"
                    mode.fault(failure)
        except BaseException as error:
            failure = str(error)
            raise
        finally:
            if process is not None:
                process.stdin.close()
                try:
                    code = process.wait(timeout=30)
                    if code and failure is None:
                        failure = f"worker_exit_{code}"
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    failure = failure or "worker_shutdown_timeout"
                process.stdout.close()
            if server is not None:
                server.stop(0).wait()
            validation = validate_barbell_trace(
                rows, confirmations, config.mode.motion_seconds
            )
            if failure is None and not validation["passed"]:
                failure = "trace_validation_failed"
            (out / "validation.json").write_text(json.dumps(validation, indent=2))
            result = dict(
                provenance,
                completed=failure is None and mode.phase == BarbellPhase.COMPLETE,
                stop_reason=failure,
                elapsed_s=clock.now,
                bar_mass_kg=config.bar_mass_kg,
                simulated_hand_devices=True,
                real_hand_grpc=True,
                confirmations=confirmations,
                phases=events,
                contact_samples=sum(
                    row["barbell/unintended_contact_force_n"] > 1 for row in rows
                ),
                unsupported_during_motion=validation["unsupported_during_motion"],
            )
            (out / "result.json").write_text(json.dumps(result, indent=2))
            (out / "hand_commands.json").write_text(
                json.dumps(devices.writes, indent=2)
            )
    if not result["completed"]:
        raise RuntimeError(f"Barbell evaluation failed: {failure}; see {out}")
    return result
