"""Synthetic checks for caller behavior, independent of robot YAML values."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bello_mujoco.hardware import gmr_retarget as module
from bello_mujoco.hardware.reference import ReferenceDiscontinuity, ReferenceUnavailable


@pytest.fixture
def boundary(monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body name="base"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body name="arm"><joint name="hinge" range="-90 90"/>
        <geom type="sphere" size=".1" mass="1"/></body>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    profile = SimpleNamespace(
        source_human="synthetic",
        target_robot="synthetic",
        source_body_names=("pelvis",),
        initial_joint_positions=(("hinge", -0.2),),
    )

    class UpstreamSignature:
        def __init__(
            self,
            src_human,
            tgt_robot,
            actual_human_height,
            solver,
            damping,
            verbose,
            use_velocity_limit,
        ):
            self.model = model
            self.robot_root_name = "base"
            self.human_root_name = "pelvis"
            self.max_iter = 10
            self.calls = 0
            self.initial_poses = []
            self.bad_output = False
            self.configuration = SimpleNamespace(data=data, update=self.update)

        def update(self, qpos):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

        def update_targets(self, targets):
            self.scaled_human_data = targets

        def retarget(self, targets):
            self.calls += 1
            self.initial_poses.append(data.qpos.copy())
            data.qpos[2] = -0.2
            data.qpos[-1] = 0.4
            if self.bad_output:
                data.qpos[0] = np.nan
            return data.qpos

    monkeypatch.setattr(
        module, "motion_task_definition", lambda asset: SimpleNamespace(gmr=profile)
    )
    monkeypatch.setattr(module, "_configure_pinned_gmr_profile", lambda *args: None)
    monkeypatch.setattr(
        module, "import_pinned_gmr_retargeter", lambda: UpstreamSignature
    )
    env = SimpleNamespace(
        model=SimpleNamespace(
            robot=SimpleNamespace(asset="synthetic"),
            xml_path="unused",
            mj_model=model,
        )
    )
    return module.GmrRetargeter(env)


def frame(sequence=1, timestamp=1_000_000_000):
    return module.HumanTargetFrame(
        transforms={
            "pelvis": (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        },
        sequence=sequence,
        source_timestamp_ns=timestamp,
        source_age_ms=0,
        received_at=0.0,
    )


def test_one_solve_and_one_time_seed_without_ground_or_posture_correction(boundary):
    output = boundary.standing(frame())
    engine = boundary.retargeter
    assert engine.calls == 1
    assert engine.max_iter == 10
    np.testing.assert_allclose(engine.initial_poses[0][:3], [1, 2, 3])
    assert engine.initial_poses[0][-1] == -0.2
    assert output.qpos[2] == -0.2
    assert output.qpos[-1] == 0.4
    live = boundary.retarget(frame())
    assert engine.calls == 2
    assert engine.initial_poses[-1][-1] == -0.2
    assert boundary.retarget(frame()) is live
    assert engine.calls == 2
    boundary.retarget(frame(2, 1_100_000_000))
    assert engine.calls == 3
    assert live.qpos[2] == -0.2


@pytest.mark.parametrize(
    "sequence,timestamp", [(0, 1_100_000_000), (2, 900_000_000), (2, 2_000_000_000)]
)
def test_rejects_invalid_source_progression(boundary, sequence, timestamp):
    boundary.retarget(frame())
    with pytest.raises(ReferenceDiscontinuity):
        boundary.retarget(frame(sequence, timestamp))
    assert boundary.retargeter.calls == 1


def test_nonfinite_solver_output_rolls_back_and_rejects(boundary):
    boundary.retarget(frame())
    previous = boundary.retargeter.configuration.data.qpos.copy()
    boundary.retargeter.bad_output = True
    with pytest.raises(ReferenceDiscontinuity):
        boundary.retarget(frame(2, 1_100_000_000))
    np.testing.assert_array_equal(boundary.retargeter.configuration.data.qpos, previous)
    assert boundary._last_sequence == 1


def test_external_rejection_preserved_for_standing_and_live(boundary):
    def reject(reference):
        raise ReferenceUnavailable("external safety screen")

    boundary.validate_reference = reject
    with pytest.raises(ReferenceUnavailable, match="external safety"):
        boundary.standing(frame())
    with pytest.raises(ReferenceUnavailable, match="external safety"):
        boundary.retarget(frame())
    assert boundary._last_sequence is None


def test_missing_source_joint_rejected_before_solving(boundary):
    invalid = frame()
    invalid.transforms.clear()
    with pytest.raises(ReferenceUnavailable, match="missing"):
        boundary.retarget(invalid)
    assert boundary.retargeter.calls == 0


def test_raw_live_source_starts_without_standing_sample_and_clears_outage(boundary):
    import queue
    import time

    packets = queue.Queue()

    class Stream:
        def wait_next(self, *, after_sequence, timeout_seconds):
            try:
                item = packets.get(timeout=timeout_seconds)
            except queue.Empty:
                return None
            if isinstance(item, Exception):
                raise item
            return item

    with module.LiveGmrReferenceSource(Stream(), boundary) as source:
        assert boundary.retargeter.calls == 0
        packets.put(frame())
        assert source.wait_until_ready(1).sequence == 1
        packets.put(ReferenceUnavailable("body tracking lost"))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                source.latest()
            except ReferenceUnavailable:
                break
            time.sleep(0.005)
        else:
            pytest.fail("outage kept publishing the previous valid pose")
        packets.put(ReferenceDiscontinuity("clock restarted"))
        packets.put(frame(sequence=1, timestamp=10))
        assert source.wait_until_ready(1).sequence == 1
        assert boundary.retargeter.calls == 2
