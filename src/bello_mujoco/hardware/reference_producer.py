"""Independent XRoboToolkit/GMR producer of latest URDF joint references; no motor IO."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math

import grpc
import numpy as np

from bello_mujoco.hardware import reference_pb2 as pb
from bello_mujoco.hardware.reference import ReferenceUnavailable, RobotReferenceSource
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.gmr_retarget import GmrLiveRetargetConfig
from bello_mujoco.robot import AssetName


@dataclass(frozen=True)
class ProducerConfig:
    asset: AssetName
    listen: str
    xrobot: PicoGrpcConfig
    retarget: GmrLiveRetargetConfig


class ReferenceService:
    def __init__(
        self,
        robot,
        source: RobotReferenceSource,
        joint_names,
        qpos_indices,
        qvel_indices,
        nq: int,
        nv: int,
    ):
        self.robot = robot
        self.source = source
        self.names = tuple(joint_names)
        self.qpos_indices = np.asarray(qpos_indices)
        self.qvel_indices = np.asarray(qvel_indices)
        self.nq, self.nv = nq, nv
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("reference producer requires unique held joints")
        for indices, size in ((self.qpos_indices, nq), (self.qvel_indices, nv)):
            if (
                indices.shape != (len(self.names),)
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or np.any(indices >= size)
            ):
                raise ValueError("invalid reference joint indices")

    def get(self, request, context):
        if request.robot != self.robot:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "reference robot mismatch")
        try:
            reference = self.source.latest()
        except ReferenceUnavailable:
            return pb.GetReferenceResponse(robot=self.robot, available=False)
        except RuntimeError as error:
            context.abort(grpc.StatusCode.INTERNAL, str(error))
        qpos, qvel = np.asarray(reference.qpos), np.asarray(reference.qvel)
        age = reference.age_ms()
        if (
            qpos.shape != (self.nq,)
            or qvel.shape != (self.nv,)
            or not np.all(np.isfinite(qpos))
            or not np.all(np.isfinite(qvel))
            or age is None
            or not math.isfinite(age)
            or age < 0
            or reference.sequence < 0
        ):
            context.abort(
                grpc.StatusCode.INTERNAL, "invalid retargeted joint reference"
            )
        return pb.GetReferenceResponse(
            robot=self.robot,
            available=True,
            sequence=reference.sequence,
            source_age_ms=age,
            joint_names=self.names,
            position=qpos[self.qpos_indices].tolist(),
            velocity=qvel[self.qvel_indices].tolist(),
        )

    def handler(self):
        return grpc.method_handlers_generic_handler(
            "bello.reference.v1.ReferenceService",
            {
                "GetReference": grpc.unary_unary_rpc_method_handler(
                    self.get,
                    request_deserializer=pb.GetReferenceRequest.FromString,
                    response_serializer=pb.GetReferenceResponse.SerializeToString,
                )
            },
        )


def run(config: ProducerConfig) -> None:
    from bello_mujoco.envs.locomotion import BelloLocomotionEnv
    from bello_mujoco.models.compiler import physical_model_config
    from bello_mujoco.hardware.xrobot_stream import XRobotGrpcStream
    from bello_mujoco.hardware.gmr_retarget import (
        GmrRetargeter,
        LiveGmrReferenceSource,
    )
    from bello_mujoco.robot_tasks import motion_task_definition

    env = BelloLocomotionEnv.load(
        physical_model_config(asset=config.asset), load_mjx=False
    )
    model = env.model
    indices = model.hold_ctrl_indices
    profile = motion_task_definition(config.asset).gmr.for_xrobot()
    with XRobotGrpcStream(config.xrobot) as stream:
        retargeter = GmrRetargeter(env, config.retarget, profile=profile)
        with LiveGmrReferenceSource(stream, retargeter) as source:
            service = ReferenceService(
                config.asset,
                source,
                [model.actuator_joint_names[i] for i in indices],
                model.actuator_qpos_indices[indices],
                model.actuator_qvel_indices[indices],
                model.mj_model.nq,
                model.mj_model.nv,
            )
            with ThreadPoolExecutor(max_workers=4) as executor:
                server = grpc.server(
                    executor,
                    options=(
                        ("grpc.so_reuseport", 0),
                        ("grpc.max_receive_message_length", 64 * 1024),
                    ),
                )
                server.add_generic_rpc_handlers((service.handler(),))
                if not server.add_insecure_port(config.listen):
                    raise RuntimeError("cannot bind reference producer")
                server.start()
                print(
                    f"Joint references at {config.listen}; no motor control.",
                    flush=True,
                )
                try:
                    server.wait_for_termination()
                except KeyboardInterrupt:
                    pass
                finally:
                    server.stop(0).wait(timeout=2.0)
