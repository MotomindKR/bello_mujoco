"""Adapt neutral-model SMPL-X pose parameters to Bello's GMR boundary."""

import numpy as np

from bello_mujoco.hardware.reference import ReferenceUnavailable
from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.hardware.smplx_grpc import SmplxGrpcConfig, SmplxGrpcStream
from bello_mujoco.hardware.smplx_kinematics import SmplxKinematics
from bello_mujoco.hardware.smplx_retarget import (
    SmplxTargetFrame,
    load_smplx_standing_sample,
)


class SmplxTargetGrpcStream:
    def __init__(self, config: SmplxGrpcConfig, source_body_names: tuple[str, ...]):
        self.config = config
        self.names = source_body_names
        unknown = set(self.names) - set(SMPLX_BODY_JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown SMPL-X body joints: {sorted(unknown)}")
        self.indices = [SMPLX_BODY_JOINT_NAMES.index(name) for name in self.names]
        self.kinematics = SmplxKinematics(config.neutral_model_path)
        self.transport = SmplxGrpcStream(config)
        self._standing = load_smplx_standing_sample(
            config.neutral_model_path, self.names
        )

    def __enter__(self):
        self.transport.__enter__()
        return self

    def __exit__(self, *exception):
        return self.transport.__exit__(*exception)

    def standing_sample(self):
        return self._standing

    def wait_next(self, *, after_sequence, timeout_seconds):
        frame = self.transport.wait_next(
            after_sequence=after_sequence, timeout_seconds=timeout_seconds
        )
        if frame is None:
            return None
        if not frame.body_valid:
            raise ReferenceUnavailable(
                "SMPL-X body tracking is invalid or uncalibrated"
            )
        pose = self.kinematics.forward(
            frame.rotations, frame.root_translation, world=True
        )
        transforms = {
            name: (
                np.array(pose.positions[i], copy=True),
                np.array(pose.quaternions[i], copy=True),
            )
            for name, i in zip(self.names, self.indices, strict=True)
        }
        return SmplxTargetFrame(
            transforms,
            frame.sequence,
            frame.source_timestamp_ns,
            int(frame.source_age_ms),
            frame.received_at,
        )
