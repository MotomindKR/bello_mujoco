"""Source motion conversion and native SMPL-X export through pinned SOMA-X APIs."""

from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.motion.bvh import BvhMotion, lafan1_body_targets
from bello_mujoco.motion.smplx import SmplxMotionReader
from bello_mujoco.sonic.corpus import file_sha256


SOMA_ASSET_FILES = (
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "SMPLX/SOMA_wrap.obj",
    "SMPLX/base_body.obj",
)


_BODY_JOINTS = {
    "Hips": "pelvis",
    "Spine1": "spine1",
    "Spine2": "spine2",
    "Chest": "spine3",
    "Neck1": "neck",
    "Head": "head",
    **{
        side + soma: prefix + native
        for side, prefix in (("Left", "left_"), ("Right", "right_"))
        for soma, native in (
            ("Shoulder", "collar"),
            ("Arm", "shoulder"),
            ("ForeArm", "elbow"),
            ("Hand", "wrist"),
            ("Leg", "hip"),
            ("Shin", "knee"),
            ("Foot", "ankle"),
            ("ToeBase", "foot"),
        )
    },
}


def soma_asset_paths(config, model_directory):
    if config.asset_directory is None:
        raise ValueError("SOMA conversion requires an explicit asset_directory")
    paths = tuple(config.asset_directory / name for name in SOMA_ASSET_FILES) + (
        model_directory / "SMPLX_NEUTRAL.npz",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


@dataclass(frozen=True)
class SomaConversionConfig:
    asset_directory: Path | None
    device: Literal["cpu", "cuda"]
    cpu_threads: int
    batch_size: int

    def __post_init__(self):
        for name in ("batch_size", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"SOMA {name} must be a positive integer")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("SOMA device must be cpu or cuda")


@dataclass(frozen=True)
class SomaLafanConfig(SomaConversionConfig):
    fit_iterations: int
    fit_learning_rate: float
    pose_regularization: float

    def __post_init__(self):
        super().__post_init__()
        if type(self.fit_iterations) is not int or self.fit_iterations <= 0:
            raise ValueError("SOMA fit_iterations must be a positive integer")
        for name in ("fit_learning_rate", "pose_regularization"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SOMA {name} must be positive and finite")


def soma_local_rotations(names, parents, reference, targets):
    """Compose anatomical motion with SOMA's bind frames; leave fingers at rest."""
    if names[0] != "Root" or parents[0] != 0:
        raise ValueError("SOMA requires its identity virtual Root")
    if missing := set(_BODY_JOINTS) - set(names):
        raise ValueError(f"SOMA rig is missing body joints: {sorted(missing)}")
    frames = len(targets["pelvis"][0])
    world = []
    for joint, name in enumerate(names):
        parent = parents[joint]
        if joint == 0:
            rotation = np.broadcast_to(np.eye(3), (frames, 3, 3))
        elif name in _BODY_JOINTS:
            rotation = (
                Rotation.from_quat(
                    targets[_BODY_JOINTS[name]][1], scalar_first=True
                ).as_matrix()
                @ reference[joint, :3, :3]
            )
        else:
            rotation = (
                world[parent] @ reference[parent, :3, :3].T @ reference[joint, :3, :3]
            )
        world.append(rotation)
    return np.stack(
        [world[parents[j]].swapaxes(-1, -2) @ world[j] for j in range(1, len(names))],
        axis=1,
    )


class SomaMotionConverter:
    def __init__(self, config: SomaConversionConfig, model_directory: Path):
        import torch
        from soma import SOMALayer
        from soma.smpl import SMPLFamilyTopologyBridge, SMPLXLayer

        paths = soma_asset_paths(config, model_directory)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SOMA device=cuda requires an available CUDA device")
        torch.set_num_threads(config.cpu_threads)
        self.config = config
        self.model_directory = model_directory
        self.reader = SmplxMotionReader(model_directory)
        self.soma = SOMALayer(
            config.asset_directory,
            identity_model_type="soma",
            device=config.device,
            mode="torch",
            enable_procedural_transforms=False,
            correctives_model_path=None,
        ).requires_grad_(False)
        self.native = SMPLXLayer(
            config.asset_directory,
            model_path=model_directory / "SMPLX_NEUTRAL.npz",
            num_betas=16,
            device=config.device,
            mode="torch",
        ).requires_grad_(False)
        self.identity = torch.zeros(
            1, self.soma.identity_model.num_identity_coeffs, device=config.device
        )
        self.betas = torch.zeros(1, 16, device=config.device)
        with torch.no_grad():
            self.soma.prepare_identity(self.identity)
            self.native.prepare_identity(self.betas)
        self.bridge = SMPLFamilyTopologyBridge(self.soma, self.native)
        self.provenance = {
            "soma_version": importlib.metadata.version("py-soma-x"),
            "human_asset_sha256": {str(p.name): file_sha256(p) for p in paths},
            "smplx_identity": "neutral-zero-betas",
        }

    def export(self, motion, destination, rotations, translations, scale, targets):
        """Transfer an absolute SOMA pose sequence to a verified native export."""
        import torch
        from soma.io import save_soma_npz
        from soma.smpl import transfer_smpl_family_pose_parameters

        native_poses, native_translations, native_positions = [], [], []
        source_squared_error, source_count = 0.0, 0
        vertex_squared_error, vertex_count = 0.0, 0
        names = self.soma.public_joint_names
        joint_ids = [names.index(name) for name in _BODY_JOINTS]
        source = np.stack([targets[name][0] for name in _BODY_JOINTS.values()], axis=1)
        for start in range(0, len(rotations), self.config.batch_size):
            stop = min(len(rotations), start + self.config.batch_size)
            rotation = torch.as_tensor(
                rotations[start:stop], dtype=torch.float32, device=self.config.device
            )
            translation = torch.as_tensor(
                translations[start:stop], dtype=torch.float32, device=self.config.device
            )
            with torch.no_grad():
                transfer = transfer_smpl_family_pose_parameters(
                    self.soma,
                    self.native,
                    rotation,
                    source_identity_coeffs=self.identity,
                    target_identity_coeffs=self.betas,
                    source_root_translation=translation,
                    source_pose2rot=False,
                    source_absolute_pose=True,
                    source_prepare_kwargs={"scale_params": scale},
                    source_pose_kwargs={"apply_correctives": False},
                    topology_bridge=self.bridge,
                )
                soma_positions = (
                    self.soma.pose(
                        rotation,
                        transl=translation,
                        pose2rot=False,
                        absolute_pose=True,
                        apply_correctives=False,
                        fk_only=True,
                    )["transforms"][:, joint_ids, :3, 3]
                    .cpu()
                    .numpy()
                )
                source_squared_error += float(
                    np.square(soma_positions - source[start:stop]).sum()
                )
                source_count += soma_positions.shape[0] * soma_positions.shape[1]
            matrices = transfer.rotations.cpu().numpy()
            serialized = Rotation.from_matrix(matrices.reshape(-1, 3, 3))
            native_poses.append(serialized.as_rotvec().reshape(len(rotation), -1, 3))
            with torch.no_grad():
                native_positions.append(
                    self.native.pose(
                        torch.as_tensor(
                            serialized.as_matrix().reshape(matrices.shape),
                            dtype=torch.float32,
                            device=self.config.device,
                        ),
                        pose2rot=False,
                        absolute_pose=True,
                        global_translation=transfer.root_translation,
                        fk_only=True,
                    )["joints"][:, :22]
                    .cpu()
                    .numpy()
                )
            native_translations.append(
                transfer.root_translation.cpu().numpy()
                - self.native.bind_pose_world[0, :3, 3].cpu().numpy()
            )
            vertex_squared_error += float(
                transfer.per_vertex_error.double().square().sum().cpu()
            )
            vertex_count += transfer.per_vertex_error.numel()
            print(
                json.dumps(
                    {
                        "sequence": destination.stem,
                        "phase": "soma_to_smplx",
                        "frames": stop,
                        "total_frames": len(rotations),
                    }
                ),
                flush=True,
            )
        poses = np.concatenate(native_poses)
        trans = np.concatenate(native_translations)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(
                stream,
                surface_model_type="smplx",
                gender="neutral",
                mocap_frame_rate=1 / motion.timestep,
                root_orient=poses[:, 0],
                pose_body=poses[:, 1:22].reshape(len(poses), 63),
                poses=poses,
                trans=trans,
                betas=self.betas.cpu().numpy()[0],
            )
        soma_path = destination.with_name(destination.stem + ".soma.npz")
        save_soma_npz(
            soma_path,
            rotations,
            translations,
            joint_names=names[1:],
            identity_model_type="soma",
            identity_coeffs=self.identity,
            scale_params=scale,
            extra_arrays={"fps": np.float64(1 / motion.timestep)},
        )
        position, _, _ = self.reader.read(destination, 1 / motion.timestep)
        expected = np.concatenate(native_positions)
        if position.shape != expected.shape or not np.allclose(
            position, expected, atol=2e-5, rtol=1e-6
        ):
            raise ValueError("native SMPL-X serialization changed SOMA-X joint FK")
        native_source = np.stack(
            [targets[name][0] for name in SMPLX_BODY_JOINT_NAMES], axis=1
        )
        return {
            **self.provenance,
            "soma_source_joint_rms_m": math.sqrt(source_squared_error / source_count),
            "smplx_source_joint_rms_m": float(
                np.sqrt(np.square(position - native_source).sum(-1).mean())
            ),
            "smplx_vertex_rms_m": math.sqrt(vertex_squared_error / vertex_count),
            "smplx_serialization_max_joint_error_m": float(
                np.abs(position - expected).max()
            ),
            "soma_sha256": file_sha256(soma_path),
        }


class SomaLafanConverter(SomaMotionConverter):
    def _prepare_skeleton(self, targets):
        import torch

        layer = self.soma
        with torch.no_grad():
            layer.prepare_identity(self.identity)
        names = layer.public_joint_names
        parents = layer.public_joint_parent_ids.cpu().numpy()
        bind = layer.public_bind_transforms_world()[0].cpu().numpy()
        scales = np.ones(len(layer.scale_param_names), dtype=np.float32)
        for index, name in enumerate(layer.scale_param_names):
            if name not in _BODY_JOINTS:
                continue
            joint = names.index(name)
            parent = parents[joint]
            if names[parent] not in _BODY_JOINTS:
                continue
            a, b = (_BODY_JOINTS[n] for n in (name, names[parent]))
            desired = np.linalg.norm(targets[a][0][0] - targets[b][0][0])
            length = np.linalg.norm(bind[joint, :3, 3] - bind[parent, :3, 3])
            if min(desired, length) <= 1e-6:
                raise ValueError(f"degenerate SOMA/source bone: {name}")
            scales[index] = desired / length
        scale = torch.as_tensor(scales[None], device=self.config.device)
        with torch.no_grad():
            layer.prepare_identity(self.identity, scale_params=scale)
        return names, parents, scale

    def _fit_batch(self, initial, translation, target, joint_ids):
        import torch

        initial = torch.as_tensor(
            initial, dtype=torch.float32, device=self.config.device
        )
        target = torch.as_tensor(target, dtype=torch.float32, device=self.config.device)
        translation = torch.as_tensor(
            translation, dtype=torch.float32, device=self.config.device
        ).clone()
        joint_ids = torch.as_tensor(joint_ids, device=self.config.device)
        with torch.no_grad():
            seeded = self.soma.pose(
                initial,
                transl=translation,
                pose2rot=False,
                absolute_pose=True,
                apply_correctives=False,
                fk_only=True,
            )["transforms"][:, joint_ids, :3, 3]
            translation += (target - seeded).mean(dim=1)
        translation = torch.nn.Parameter(translation)
        delta = torch.nn.Parameter(
            torch.zeros(len(initial), len(joint_ids), 3, device=self.config.device)
        )
        optimizer = torch.optim.Adam(
            [delta, translation], lr=self.config.fit_learning_rate
        )

        def pose():
            x, y, z = delta.unbind(-1)
            zero = torch.zeros_like(x)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
            rotation = initial.index_copy(
                1,
                joint_ids - 1,
                initial[:, joint_ids - 1]
                @ torch.matrix_exp(skew.reshape(*delta.shape[:2], 3, 3)),
            )
            joints = self.soma.pose(
                rotation,
                transl=translation,
                pose2rot=False,
                absolute_pose=True,
                apply_correctives=False,
                fk_only=True,
            )["transforms"][:, joint_ids, :3, 3]
            return rotation, joints

        for _ in range(self.config.fit_iterations):
            optimizer.zero_grad()
            _, joints = pose()
            loss = (
                joints - target
            ).square().mean() + self.config.pose_regularization * delta.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite SOMA body fit")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            rotation, joints = pose()
            squared_error = (joints - target).square().sum(-1)
        return rotation.detach(), translation.detach(), squared_error.cpu().numpy()

    def convert(self, motion: BvhMotion, destination: Path):
        targets = lafan1_body_targets(motion)
        names, parents, scale = self._prepare_skeleton(targets)
        initial = soma_local_rotations(
            names, parents, self.soma.t_pose_world.cpu().numpy(), targets
        )
        joint_ids = [names.index(name) for name in _BODY_JOINTS]
        source = np.stack([targets[name][0] for name in _BODY_JOINTS.values()], axis=1)
        rotations, translations = [], []
        for start in range(0, len(source), self.config.batch_size):
            stop = min(len(source), start + self.config.batch_size)
            rotation, translation, _ = self._fit_batch(
                initial[start:stop],
                targets["pelvis"][0][start:stop],
                source[start:stop],
                joint_ids,
            )
            rotations.append(rotation.cpu().numpy())
            translations.append(translation.cpu().numpy())
        return self.export(
            motion,
            destination,
            np.concatenate(rotations),
            np.concatenate(translations),
            scale,
            targets,
        )


def soma_bvh_rotations(motion, bind, names, parents, reference):
    """Express every BVH joint's bind-relative motion in SOMA's public frames."""
    if motion.names != bind.names or motion.parents != bind.parents:
        raise ValueError("SOMA source and bind hierarchies differ")
    if tuple(names) != bind.names or tuple(parents[1:]) != bind.parents[1:]:
        raise ValueError("BVH must match the SOMA public hierarchy")
    if names[0] != "Root" or parents[0] != 0:
        raise ValueError("SOMA requires its identity virtual Root")
    if not np.allclose(bind.position, bind.position[:1]) or not np.allclose(
        bind.quaternion, bind.quaternion[:1]
    ):
        raise ValueError("SOMA bind must have a constant rest pose")
    positions, quaternions = motion.world_transforms()
    _, bind_quaternions = bind.world_transforms()
    source_world = (
        Rotation.from_quat(quaternions.reshape(-1, 4), scalar_first=True)
        .as_matrix()
        .reshape(*quaternions.shape[:2], 3, 3)
    )
    bind_world = Rotation.from_quat(bind_quaternions[0], scalar_first=True).as_matrix()
    up = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    world = source_world @ bind_world.swapaxes(-1, -2) @ up @ reference[:, :3, :3]
    # Root is virtual in SOMA. Its source motion is already present in Hips.
    world[:, 0] = np.eye(3)
    local = world[:, parents[1:]].swapaxes(-1, -2) @ world[:, 1:]
    targets = {
        native: (positions[:, names.index(name)], quaternions[:, names.index(name)])
        for name, native in _BODY_JOINTS.items()
    }
    return local, positions[:, names.index("Hips")], targets


class SomaBonesConverter(SomaMotionConverter):
    def __init__(self, config, model_directory, bind):
        import torch

        super().__init__(config, model_directory)
        self.bind = bind
        self.names = self.soma.public_joint_names
        self.parents = self.soma.public_joint_parent_ids.cpu().numpy()
        if (
            tuple(self.names) != bind.names
            or tuple(self.parents[1:]) != bind.parents[1:]
        ):
            raise ValueError("BONES bind does not match the SOMA public hierarchy")
        reference = self.soma.public_bind_transforms_world()[0].cpu().numpy()
        positions, _ = bind.world_transforms()
        scales = []
        for name in self.soma.scale_param_names:
            joint = self.names.index(name)
            parent = self.parents[joint]
            length = np.linalg.norm(reference[joint, :3, 3] - reference[parent, :3, 3])
            desired = np.linalg.norm(positions[0, joint] - positions[0, parent])
            if min(length, desired) <= 1e-6:
                raise ValueError(f"degenerate SOMA/source bone: {name}")
            scales.append(desired / length)
        self.scale = torch.as_tensor(
            np.asarray(scales, dtype=np.float32)[None], device=config.device
        )
        with torch.no_grad():
            self.soma.prepare_identity(self.identity, scale_params=self.scale)

    def convert(self, motion, destination):
        rotations, translation, targets = soma_bvh_rotations(
            motion,
            self.bind,
            self.names,
            self.parents,
            self.soma.t_pose_world.cpu().numpy(),
        )
        return self.export(
            motion, destination, rotations, translation, self.scale, targets
        )
