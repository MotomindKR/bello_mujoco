"""Offline BVH parsing and LAFAN1 conversion to canonical human target frames."""

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BvhMotion:
    """Local WXYZ rotations and metre translations in the source's Y-up frame."""

    names: tuple[str, ...]
    parents: tuple[int, ...]
    position: np.ndarray
    quaternion: np.ndarray
    timestep: float

    def __post_init__(self):
        count = len(self.names)
        if not count or len(set(self.names)) != count:
            raise ValueError("BVH joint names must be nonempty and unique")
        if (
            len(self.parents) != count
            or self.parents[0] != -1
            or any(
                not 0 <= parent < index
                for index, parent in enumerate(self.parents[1:], 1)
            )
        ):
            raise ValueError("BVH must have one root and topologically ordered parents")
        if not math.isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError("BVH frame time must be positive and finite")
        position = np.array(self.position, dtype=np.float64, copy=True)
        quaternion = np.array(self.quaternion, dtype=np.float64, copy=True)
        if (
            position.ndim != 3
            or position.shape[1:] != (count, 3)
            or not len(position)
            or quaternion.shape != position.shape[:2] + (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=1e-6)
        ):
            raise ValueError("invalid BVH motion arrays or quaternion norms")
        position.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)

    @property
    def duration(self):
        return (len(self.position) - 1) * self.timestep

    def resample(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("BVH output fps must be positive and finite")
        frame = np.arange(math.floor(self.duration * fps) + 1) / fps / self.timestep
        lower = np.floor(frame).astype(int)
        upper = np.minimum(lower + 1, len(self.position) - 1)
        alpha = (frame - lower)[:, None, None]
        position = (1 - alpha) * self.position[lower] + alpha * self.position[upper]
        a, b = self.quaternion[lower], self.quaternion[upper]
        dot = np.sum(a * b, axis=-1, keepdims=True)
        b = np.where(dot < 0, -b, b)
        angle = np.arccos(np.clip(np.abs(dot), 0, 1))
        sine = np.sin(angle)
        quaternion = np.where(
            sine > 1e-7,
            (np.sin((1 - alpha) * angle) * a + np.sin(alpha * angle) * b)
            / np.maximum(sine, 1e-7),
            (1 - alpha) * a + alpha * b,
        )
        quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
        return BvhMotion(self.names, self.parents, position, quaternion, 1 / fps)

    def world_transforms(self):
        """Return Z-up world positions in metres and WXYZ body orientations."""
        frames, joints = self.position.shape[:2]
        positions = np.empty_like(self.position)
        quaternions = np.empty_like(self.quaternion)
        rotations = []
        up = Rotation.from_euler("x", 90, degrees=True)
        for joint, parent in enumerate(self.parents):
            local = Rotation.from_quat(self.quaternion[:, joint], scalar_first=True)
            if parent < 0:
                positions[:, joint] = up.apply(self.position[:, joint])
                rotations.append(up * local)
            else:
                positions[:, joint] = positions[:, parent] + rotations[parent].apply(
                    self.position[:, joint]
                )
                rotations.append(rotations[parent] * local)
            quaternions[:, joint] = rotations[joint].as_quat(scalar_first=True)
        assert positions.shape == (frames, joints, 3)
        return positions, quaternions


def read_bvh(path: Path) -> BvhMotion:
    """Read the Y-up, centimetre BVH convention used by LAFAN1 and SOMA."""
    text = Path(path).read_text()
    sections = re.split(r"(?m)^\s*MOTION\s*$", text, maxsplit=1)
    if len(sections) != 2:
        raise ValueError("BVH requires a MOTION section")
    tokens = re.findall(r"[{}]|[^\s{}]+", sections[0])
    cursor = 0

    def take(expected=None):
        nonlocal cursor
        if cursor == len(tokens):
            raise ValueError("truncated BVH hierarchy")
        value = tokens[cursor]
        cursor += 1
        if expected is not None and value != expected:
            raise ValueError(f"BVH expected {expected!r}, got {value!r}")
        return value

    names, parents, offsets, channels = [], [], [], []

    def joint(parent, kind):
        take(kind)
        name = take()
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(None)
        channels.append(())
        take("{")
        seen = set()
        while True:
            if cursor == len(tokens):
                raise ValueError("truncated BVH joint")
            if tokens[cursor] == "}":
                break
            token = tokens[cursor]
            if token == "JOINT":
                joint(index, "JOINT")
            elif token == "End":
                take("End")
                take("Site")
                take("{")
                take("OFFSET")
                for _ in range(3):
                    float(take())
                take("}")
            elif token in {"OFFSET", "CHANNELS"}:
                if token in seen:
                    raise ValueError(f"duplicate BVH {token} for {name}")
                seen.add(take())
                if token == "OFFSET":
                    offsets[index] = [float(take()) for _ in range(3)]
                else:
                    size = int(take())
                    if not 0 <= size <= 6:
                        raise ValueError(
                            "BVH channel count must be between zero and six"
                        )
                    channels[index] = tuple(take() for _ in range(size))
            else:
                raise ValueError(f"unexpected BVH hierarchy token: {token}")
            if cursor == len(tokens):
                raise ValueError("truncated BVH joint")
        take("}")
        if offsets[index] is None:
            raise ValueError(f"missing BVH offset for {name}")

    take("HIERARCHY")
    joint(-1, "ROOT")
    if cursor != len(tokens):
        raise ValueError("unexpected content after BVH root")
    lines = [line.strip() for line in sections[1].splitlines() if line.strip()]
    if (
        len(lines) < 3
        or not lines[0].startswith("Frames:")
        or not lines[1].startswith("Frame Time:")
    ):
        raise ValueError("invalid BVH motion header")
    frames = int(lines[0].split(":", 1)[1])
    timestep = float(lines[1].split(":", 1)[1])
    values = np.loadtxt(lines[2:], ndmin=2)
    if values.shape != (frames, sum(map(len, channels))):
        raise ValueError("BVH frame/channel count differs from its header")
    position = np.broadcast_to(np.asarray(offsets), (frames, len(names), 3)).copy()
    quaternion = np.zeros((frames, len(names), 4))
    quaternion[..., 0] = 1
    cursor = 0
    allowed = {axis + kind for axis in "XYZ" for kind in ("position", "rotation")}
    for index, labels in enumerate(channels):
        if len(set(labels)) != len(labels) or not set(labels) <= allowed:
            raise ValueError(f"invalid BVH channels for {names[index]}")
        block = values[:, cursor : cursor + len(labels)]
        cursor += len(labels)
        order, angles = "", []
        for column, label in enumerate(labels):
            if label.endswith("position"):
                # These sources store translated joint coordinates, replacing OFFSET.
                position[:, index, "XYZ".index(label[0])] = block[:, column]
            else:
                order += label[0]
                angles.append(block[:, column])
        if order:
            quaternion[:, index] = Rotation.from_euler(
                order, np.stack(angles, axis=-1), degrees=True
            ).as_quat(scalar_first=True)
    return BvhMotion(tuple(names), tuple(parents), position / 100, quaternion, timestep)


# LAFAN's X-along-bone frames expressed in the canonical SMPL-X target frames.
# The paired source conventions agree with the pinned GMR LAFAN/SMPL-X profiles.
_HALF_ROOT = (-0.5, 0.5, 0.5, 0.5)
_HALF_LEG = (-0.5, 0.5, -0.5, -0.5)
_SQRT_HALF = math.sqrt(0.5)
_LAFAN_TARGETS = (
    ("pelvis", "Hips", _HALF_ROOT),
    ("spine3", "Spine2", _HALF_ROOT),
    ("left_hip", "LeftUpLeg", _HALF_LEG),
    ("right_hip", "RightUpLeg", _HALF_LEG),
    ("left_knee", "LeftLeg", _HALF_LEG),
    ("right_knee", "RightLeg", _HALF_LEG),
    ("left_foot", "LeftToe", (-_SQRT_HALF, 0.0, -_SQRT_HALF, 0.0)),
    ("right_foot", "RightToe", (-_SQRT_HALF, 0.0, -_SQRT_HALF, 0.0)),
    ("left_shoulder", "LeftArm", (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),
    ("right_shoulder", "RightArm", (0.0, 0.0, _SQRT_HALF, -_SQRT_HALF)),
    ("left_elbow", "LeftForeArm", (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),
    ("right_elbow", "RightForeArm", (0.0, 0.0, -_SQRT_HALF, _SQRT_HALF)),
    ("left_wrist", "LeftHand", (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)),
    ("right_wrist", "RightHand", (0.0, 0.0, -_SQRT_HALF, _SQRT_HALF)),
)


def lafan1_targets(motion: BvhMotion):
    """Convert named LAFAN poses, including toe-centre foot targets, for Mini GMR."""
    missing = {bone for _, bone, _ in _LAFAN_TARGETS} - set(motion.names)
    if missing:
        raise ValueError(f"LAFAN1 skeleton is missing joints: {sorted(missing)}")
    position, quaternion = motion.world_transforms()
    result = {}
    for name, bone, basis in _LAFAN_TARGETS:
        index = motion.names.index(bone)
        orientation = Rotation.from_quat(
            quaternion[:, index], scalar_first=True
        ) * Rotation.from_quat(basis, scalar_first=True)
        result[name] = (
            position[:, index],
            orientation.as_quat(scalar_first=True),
        )
    return result
