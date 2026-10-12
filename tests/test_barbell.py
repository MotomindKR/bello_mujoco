from dataclasses import replace

import pytest

from bello_mujoco.barbell import (
    BarbellMode,
    BarbellModeConfig,
    BarbellPhase as P,
    HandoffConfirmation as C,
)
from bello_mujoco.hardware.barbell_hands import BarbellHandSession
from bello_mujoco.hardware import hand_pb2 as pb
from test_hand_server import setup_controller


def mode_config():
    return BarbellModeConfig(
        initial_stand_seconds=0.2,
        reach_seconds=0.2,
        load_transfer_seconds=0.4,
        policy_blend_seconds=0.4,
        motion_seconds=0.4,
        retract_seconds=0.2,
        final_stand_seconds=0.2,
    )


def reach(mode, phase):
    for _ in range(100):
        if mode.phase == phase:
            return
        command = mode.snapshot()
        if command.awaiting:
            mode.confirm(command.awaiting, phase_id=command.phase_id)
        else:
            mode.advance(0.1)
    raise AssertionError(f"failed to reach {phase}")


def test_complete_cycle_has_confirmation_gates_and_supported_release():
    mode = BarbellMode(mode_config())
    commands = []
    events = []
    for _ in range(100):
        if mode.phase == P.COMPLETE:
            break
        command = mode.advance(0.1)
        commands.append(command)
        if command.awaiting:
            events.append(command.awaiting)
            mode.confirm(command.awaiting, phase_id=command.phase_id)
    assert mode.phase == P.COMPLETE
    assert events == [
        C.BAR_READY,
        C.GRASP_SECURED,
        C.BAR_SUPPORTED,
        C.HANDS_OPEN,
        C.BAR_CLEAR,
    ]
    loaded = [
        command
        for command in commands
        if command.phase in (P.ENTER_MOTION, P.MOTION, P.RETURN, P.SUPPORT)
    ]
    assert all(
        command.load_fraction == 1 and command.finger_closure == 1 for command in loaded
    )
    opened = [
        command
        for command in commands
        if command.phase in (P.OPEN, P.CLEAR, P.RETRACT, P.FINISH)
    ]
    assert all(
        command.load_fraction == 0 and command.finger_closure == 0 for command in opened
    )
    assert commands[-1].arm_fraction == 0
    assert commands[-1].motion_weight == 0
    assert (
        max(command.motion_time for command in commands) == mode.config.motion_seconds
    )


@pytest.mark.parametrize("phase", [P.RECEIVE, P.CLOSE, P.SUPPORT, P.OPEN, P.CLEAR])
def test_waits_never_infer_confirmations_from_elapsed_time(phase):
    mode = BarbellMode(mode_config())
    reach(mode, phase)
    before = mode.snapshot()
    after = mode.advance(100)
    assert after.phase == phase
    assert (after.arm_fraction, after.load_fraction, after.finger_closure) == (
        before.arm_fraction,
        before.load_fraction,
        before.finger_closure,
    )
    with pytest.raises(ValueError, match="generation"):
        mode.confirm(before.awaiting, phase_id=before.phase_id - 1)
    assert mode.phase == phase


@pytest.mark.parametrize(
    "phase", [P.CLOSE, P.LOAD, P.ENTER_MOTION, P.MOTION, P.RETURN, P.SUPPORT, P.UNLOAD]
)
def test_stop_never_opens_loaded_or_possibly_loaded_hands(phase):
    mode = BarbellMode(mode_config())
    reach(mode, phase)
    before = mode.advance(0.1)
    mode.stop("operator stop")
    for _ in range(20):
        command = mode.advance(0.1)
        assert command.finger_closure == before.finger_closure == 1
        assert command.load_fraction == before.load_fraction
    assert mode.snapshot().awaiting == C.BAR_SUPPORTED
    assert mode.snapshot().motion_weight == 0
    assert mode.snapshot().interrupted


def test_stop_during_entry_blend_returns_from_current_weight():
    mode = BarbellMode(mode_config())
    reach(mode, P.ENTER_MOTION)
    mode.advance(0.1)
    assert mode.snapshot().motion_weight == 0.25
    mode.stop("cancel entry")
    assert mode.advance(0.1).motion_weight == pytest.approx(0.1875)


def test_stop_during_partial_transfer_preserves_preload_until_support_confirmed():
    mode = BarbellMode(mode_config())
    reach(mode, P.LOAD)
    mode.advance(0.1)
    mode.stop("tracking disconnected")
    snapshot = mode.snapshot()
    assert 0 < snapshot.load_fraction < 1
    assert mode.advance(10).load_fraction == snapshot.load_fraction
    mode.confirm(C.BAR_SUPPORTED, phase_id=mode.phase_id)
    unload = mode.advance(0.2)
    assert unload.load_fraction == pytest.approx(snapshot.load_fraction / 2)
    mode.advance(0.2)
    assert mode.snapshot().phase == P.OPEN
    assert mode.snapshot().load_fraction == 0


def test_unloaded_stop_retracts_from_current_arm_pose():
    mode = BarbellMode(mode_config())
    reach(mode, P.REACH)
    mode.advance(0.1)
    assert mode.snapshot().arm_fraction == 0.5
    mode.stop("cancel reach")
    assert mode.advance(0.1).arm_fraction == 0.25
    reach(mode, P.COMPLETE)
    assert mode.snapshot().arm_fraction == 0


def test_fault_is_latched_and_preserves_fingers_and_load():
    mode = BarbellMode(mode_config())
    reach(mode, P.MOTION)
    mode.fault("body feedback stale")
    before = mode.snapshot()
    assert mode.advance(100) == before
    mode.stop("stop")
    assert mode.snapshot() == before
    assert before.finger_closure == before.load_fraction == 1
    with pytest.raises(ValueError):
        mode.confirm(C.BAR_SUPPORTED, phase_id=mode.phase_id)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_invalid_timing_rejected(bad):
    with pytest.raises(ValueError):
        replace(mode_config(), load_transfer_seconds=bad)
    with pytest.raises(ValueError):
        BarbellMode(mode_config()).advance(bad)


class Client:
    def __init__(self, controller):
        self.controller = controller
        self.acquisitions = 0

    def acquire(self, side):
        self.acquisitions += 1
        return self.controller.acquire(pb.Hand(side=side))

    def status(self):
        return self.controller.status()

    def command(self, command):
        return self.controller.command(command)

    def keep_alive(self, lease):
        return self.controller.keep_alive(lease)

    def release(self, lease):
        return self.controller.release(lease)


def test_session_owns_both_hands_renews_and_never_opens_on_exit():
    controller, transport, clock = setup_controller()
    client = Client(controller)
    with BarbellHandSession(client, clock=clock) as session:
        session.update(1)
        controller.tick()
        clock.now += 0.3
        session.update(1)
        controller.tick()
        assert all(hand.source == pb.APPLICATION for hand in client.status().hands)
    assert all(
        hand.source == pb.HOLD and not hand.owned for hand in client.status().hands
    )
    assert all(max(positions) < 255 for _, positions in transport.writes)
    assert client.acquisitions == 2


def test_failed_second_acquisition_rolls_back_first():
    controller, _, clock = setup_controller()
    right = controller.acquire(pb.Hand(side=pb.RIGHT))
    with pytest.raises(RuntimeError, match="owner"):
        with BarbellHandSession(Client(controller), clock=clock):
            raise AssertionError("must not enter")
    status = controller.status()
    assert not status.hands[0].owned and status.hands[1].owned
    controller.release(right)


def test_lost_lease_is_not_reacquired():
    controller, _, clock = setup_controller()
    client = Client(controller)
    session = BarbellHandSession(client, clock=clock).__enter__()
    session.update(1)
    clock.now += 2
    with pytest.raises(PermissionError):
        session.update(1)
    assert client.acquisitions == 2
    with pytest.raises(RuntimeError, match="release"):
        session.close()
    assert all(hand.source == pb.HOLD for hand in controller.status().hands)


def test_stop_while_receiving_waits_for_bar_to_clear_before_retracting():
    mode = BarbellMode(mode_config())
    reach(mode, P.RECEIVE)
    mode.stop("cancel receive")
    command = mode.advance(1)
    assert command.arm_fraction == 1
    assert command.awaiting == C.BAR_CLEAR
    assert command.finger_closure == 0
    mode.confirm(C.BAR_CLEAR, phase_id=mode.phase_id)
    assert mode.advance(0.1).arm_fraction < 1


def test_trace_audit_rejects_missing_motion_and_missing_confirmations():
    from bello_mujoco.evaluation.barbell import validate_barbell_trace

    result = validate_barbell_trace([], [], 1.0)
    assert not result["passed"]
    assert not result["full_motion"]
    assert not result["all_confirmations"]
    assert not result["unsupported_during_motion"]


def test_trace_audit_rejects_support_during_motion_and_opening_under_load():
    from bello_mujoco.evaluation.barbell import validate_barbell_trace

    row = dict(
        phase="motion",
        motion_time=1.0,
        external_support=True,
        load_fraction=1,
        finger_closure=0,
        arm_fraction=1,
    )
    row["barbell/unintended_contact_force_n"] = 0
    result = validate_barbell_trace([row], [], 1.0)
    assert not result["passed"]
    assert not result["unsupported_during_motion"]
    assert not result["closed_while_loaded"]
