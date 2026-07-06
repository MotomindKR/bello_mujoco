from bello_mujoco.contracts import ACTION_DIM, FRAME_OBS_DIM, FULL_BODY_ACTION_DIM, POLICY_JOINT_ORDER


def test_policy_contract_dimensions() -> None:
    assert ACTION_DIM == 12
    assert FULL_BODY_ACTION_DIM == 35
    assert len(POLICY_JOINT_ORDER) == 12
    assert FRAME_OBS_DIM == 114
