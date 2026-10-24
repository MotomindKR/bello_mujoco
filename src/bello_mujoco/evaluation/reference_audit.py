"""Read-only reference dynamics diagnostics under the configured contact model."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import hydra
import mujoco
import numpy as np

from bello_mujoco.configuration import (
    command_config,
    config_path,
    supermimic_training_profile,
)
from bello_mujoco.models.ratings import motor_continuous_torque
from bello_mujoco.motion.clip import load_motion_clip
from bello_mujoco.repository import capture_repository_commit
from bello_mujoco.sim import load_model
from bello_mujoco.sonic.corpus import MANIFEST, verify_corpus
from bello_mujoco.terrain import FLAT_TERRAIN
from bello_mujoco.training.tracking import to_builtin


@dataclass(frozen=True)
class ReferenceAuditConfig:
    motion_dataset: Path
    output_path: Path
    experiment: str = "bello_mini_supermimic_lafan"
    clips: tuple[str, ...] = ()
    frame_stride: int = 10
    contact_distance_m: float = 0.03
    slip_speed_m_s: float = 0.2

    def __post_init__(self):
        if type(self.frame_stride) is not int or self.frame_stride <= 0:
            raise ValueError("frame_stride must be a positive integer")
        for name in ("contact_distance_m", "slip_speed_m_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if len(set(self.clips)) != len(self.clips):
            raise ValueError("clips must be unique")


def inverse_requirements(model, data, qpos, qvel, qacc, joint_dofs, transform):
    """Inverse dynamics includes passive and configured soft-contact forces."""
    data.qpos[:], data.qvel[:], data.qacc[:] = qpos, qvel, qacc
    mujoco.mj_inverse(model, data)
    motor = np.linalg.solve(np.asarray(transform).T, data.qfrc_inverse[joint_dofs])
    return motor, data.qfrc_inverse.copy()


def ground_contact_bodies(model, data, terrain_geom_ids):
    terrain = frozenset(int(g) for g in terrain_geom_ids)
    bodies = set()
    for contact in data.contact:
        if contact.efc_address < 0:
            continue
        first, second = map(int, contact.geom)
        if first in terrain and second not in terrain:
            bodies.add(int(model.geom_bodyid[second]))
        elif second in terrain and first not in terrain:
            bodies.add(int(model.geom_bodyid[first]))
    return frozenset(bodies)


def foot_support_samples(model, data, foot_geom_ids, terrain_geom_ids, distance):
    """Near-plane material-point velocities; proximity does not imply support force."""
    samples = []
    points = np.zeros(6)
    jacobian = np.zeros((3, model.nv))
    for geom in foot_geom_ids:
        for ground in terrain_geom_ids:
            separation = mujoco.mj_geomDistance(
                model, data, int(geom), int(ground), distance, points
            )
            if separation >= distance:
                continue
            mujoco.mj_jac(
                model, data, jacobian, None, points[:3], int(model.geom_bodyid[geom])
            )
            velocity = jacobian @ data.qvel
            samples.append((points[:3].copy(), float(np.linalg.norm(velocity[:2]))))
    return samples


def support_margin(points, center):
    """Signed COM distance to a 2D convex hull; None for degenerate support."""
    points = sorted(set(tuple(p[:2]) for p in points))
    if len(points) < 3:
        return None

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result[:-1]

    hull = half(points) + half(reversed(points))
    if len(hull) < 3:
        return None
    margins = [
        cross(a, b, center) / np.linalg.norm(np.subtract(b, a))
        for a, b in zip(hull, hull[1:] + hull[:1], strict=True)
    ]
    return float(min(margins))


def audit_clip(model, clip, config):
    native = model.mj_model
    data = mujoco.MjData(native)
    velocity = np.asarray(clip.qvel, dtype=np.float64)
    acceleration = np.gradient(velocity, 1 / clip.fps, axis=0)
    transform = np.asarray(model.actuator_coordinate_transform)
    rated = motor_continuous_torque(model)
    peak = np.asarray(model.actuator_peak_force)
    free = np.flatnonzero(native.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("reference dynamics audit requires one floating base")
    root_dof = int(native.jnt_dofadr[free[0]])
    force_slice = slice(root_dof, root_dof + 3)
    torque_slice = slice(root_dof + 3, root_dof + 6)
    weight = float(native.body_mass.sum() * np.linalg.norm(native.opt.gravity))
    counts = np.zeros(native.nu, dtype=int)
    rated_counts = np.zeros(native.nu, dtype=int)
    maxima = np.zeros(native.nu)
    rows = []
    for frame in range(0, clip.num_frames, config.frame_stride):
        motor, inverse = inverse_requirements(
            native,
            data,
            np.asarray(clip.qpos[frame]),
            velocity[frame],
            acceleration[frame],
            model.actuator_qvel_indices,
            transform,
        )
        ratio = np.abs(motor) / peak
        if not np.isfinite(inverse).all() or not np.isfinite(ratio).all():
            raise RuntimeError(f"nonfinite inverse dynamics at reference frame {frame}")
        counts += ratio > 1
        rated_counts += np.abs(motor) > rated
        maxima = np.maximum(maxima, ratio)
        samples = foot_support_samples(
            native,
            data,
            model.foot_geom_ids,
            model.terrain_geom_ids,
            config.contact_distance_m,
        )
        margin = support_margin([p for p, _ in samples], data.subtree_com[0, :2])
        rows.append(
            {
                "frame": frame,
                "seconds": frame / clip.fps,
                "root_force_residual_n": float(np.linalg.norm(inverse[force_slice])),
                "root_torque_residual_nm": float(np.linalg.norm(inverse[torque_slice])),
                "max_motor_peak_ratio": float(ratio.max()),
                "max_motor_speed_rad_s": float(
                    np.abs(
                        transform @ velocity[frame, model.actuator_qvel_indices]
                    ).max()
                ),
                "max_motor_acceleration_rad_s2": float(
                    np.abs(
                        transform @ acceleration[frame, model.actuator_qvel_indices]
                    ).max()
                ),
                "max_near_foot_point_speed_m_s": max(
                    (s for _, s in samples), default=0.0
                ),
                "near_foot_points": len(samples),
                "com_near_foot_polygon_margin_m": margin,
            }
        )
    known_support = [r for r in rows if r["com_near_foot_polygon_margin_m"] is not None]
    summary = {
        "source_frames": clip.num_frames,
        "sampled_frames": len(rows),
        "sample_period_seconds": config.frame_stride / clip.fps,
        "max_root_force_residual_bodyweights": max(
            r["root_force_residual_n"] for r in rows
        )
        / weight,
        "rms_root_force_residual_bodyweights": float(
            np.sqrt(np.mean([r["root_force_residual_n"] ** 2 for r in rows]))
        )
        / weight,
        "max_motor_peak_ratio": float(maxima.max()),
        "max_motor_speed_rad_s": max(r["max_motor_speed_rad_s"] for r in rows),
        "max_motor_acceleration_rad_s2": max(
            r["max_motor_acceleration_rad_s2"] for r in rows
        ),
        "near_foot_sliding_fraction": sum(
            r["max_near_foot_point_speed_m_s"] > config.slip_speed_m_s for r in rows
        )
        / len(rows),
        "frames_with_near_foot_polygon": len(known_support),
        "com_outside_near_foot_polygon_fraction": (
            sum(r["com_near_foot_polygon_margin_m"] < 0 for r in known_support)
            / len(known_support)
            if known_support
            else None
        ),
    }
    return {
        "summary": summary,
        "motors": {
            name: {
                "inverse_over_peak_fraction": float(counts[i] / len(rows)),
                "inverse_over_rated_fraction": float(rated_counts[i] / len(rows)),
                "max_inverse_peak_ratio": float(maxima[i]),
            }
            for i, name in enumerate(model.actuator_coordinate_names)
        },
        "samples": rows,
    }


def audit_references(config):
    run = supermimic_training_profile(config.motion_dataset, name=config.experiment)
    manifest = json.loads((config.motion_dataset / MANIFEST).read_text())
    held_out = tuple(
        Path(e["file"]).stem for e in manifest["clips"] if e["split"] == "evaluation"
    )
    digest = verify_corpus(config.motion_dataset, held_out)
    members = {Path(e["file"]).stem: e for e in manifest["clips"]}
    selected = config.clips or tuple(members)
    if any(name not in members for name in selected):
        raise ValueError("clips must belong to the verified corpus")
    if config.output_path.exists():
        raise FileExistsError(config.output_path)
    model = load_model(run.asset_config, terrain=FLAT_TERRAIN, load_mjx=False)
    report = {
        "audit_config": to_builtin(config),
        "source_commit": capture_repository_commit(),
        "dataset_manifest_sha256": digest,
        "scope": "Read-only exact-reference diagnostics, not a motion feasibility certificate or dataset filter. Inverse torques include MuJoCo's configured soft contacts/passive forces; root residual is the external base wrench needed at the literal pose and differentiated acceleration. Near-ground points do not imply admissible support forces. COM polygon is a quasi-static geometric indicator; dynamic motion may leave it. Rated torque depends on duty and cooling.",
        "clips": {},
    }
    for name in selected:
        clip = load_motion_clip(
            config.motion_dataset / members[name]["file"], model, as_numpy=True
        )
        result = audit_clip(model, clip, config)
        result["split"] = members[name]["split"]
        report["clips"][name] = result
        print(json.dumps({"audited_clip": name, **result["summary"]}), flush=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="audit_reference",
)
def main(config):
    audit_references(command_config(ReferenceAuditConfig, config))


if __name__ == "__main__":
    main()
