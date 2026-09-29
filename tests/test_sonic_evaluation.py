import pytest

from bello_mujoco.evaluation.sonic_transfer import uniform_clip_summary


def row(length, completion, terminal):
    return dict(
        rendered_steps=length,
        completion_fraction=completion,
        terminal=terminal,
        episode_return=2.0,
        root_position_rms_m=0.1,
        root_orientation_rms_deg=5.0,
        joint_position_rms_rad=0.2,
    )


def test_fixed_evaluation_weights_clips_equally_not_by_duration():
    result = uniform_clip_summary(
        {"short": row(10, 1.0, False), "long": row(900, 0.5, True)}
    )
    assert result["mean_completion_fraction"] == 0.75
    assert result["mean_rendered_steps"] == 455
    assert result["failure_fraction"] == 0.5
    assert result["weighting"] == "equal_per_clip"


def test_fixed_evaluation_rejects_empty_or_nonfinite_results():
    with pytest.raises(ValueError):
        uniform_clip_summary({})
    with pytest.raises(ValueError, match="nonfinite"):
        uniform_clip_summary({"bad": row(10, float("nan"), True)})
