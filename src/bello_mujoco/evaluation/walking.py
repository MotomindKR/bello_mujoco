"""Reward-independent walking acceptance and native contact-point slip traces."""

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class WalkingAcceptanceConfig:
    enabled: bool = False
    warmup_seconds: float = 2.0
    minimum_window_seconds: float = 6.0
    velocity_rmse_max: float = 0.1
    yaw_rate_rmse_max: float = 0.2
    stance_slip_max: float = 0.05
    lateral_tilt_rms_max: float = math.radians(10.0)
    lateral_tilt_peak_max: float = math.radians(20.0)
    single_support_min: float = 0.2
    each_side_support_min: float = 0.05
    minimum_support_alternations: int = 4
    contact_debounce_seconds: float = 0.04

    def __post_init__(self):
        for name in (
            "warmup_seconds",
            "minimum_window_seconds",
            "velocity_rmse_max",
            "yaw_rate_rmse_max",
            "stance_slip_max",
            "lateral_tilt_rms_max",
            "lateral_tilt_peak_max",
            "contact_debounce_seconds",
        ):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name != "warmup_seconds" and value == 0)
            ):
                raise ValueError(
                    f"walking_acceptance.{name} must be finite and positive"
                )
        for name in ("single_support_min", "each_side_support_min"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"walking_acceptance.{name} must be in (0, 1]")
        if self.minimum_support_alternations < 1:
            raise ValueError(
                "walking_acceptance.minimum_support_alternations must be positive"
            )


@dataclass
class WalkingDiagnostics:
    timestep: float
    config: WalkingAcceptanceConfig
    samples: list[dict[str, float]] = field(default_factory=list)
    support: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    elapsed: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def record(self, sample):
        sample = dict(sample)
        invalid = not all(math.isfinite(value) for value in sample.values())
        sample = {k: float(v) if math.isfinite(v) else 0.0 for k, v in sample.items()}
        sample["invalid_state"] = float(invalid)
        raw = np.array([sample["left_contact"], sample["right_contact"]], dtype=bool)
        self.elapsed = np.where(raw != self.support, self.elapsed + self.timestep, 0.0)
        self.support = np.where(
            self.elapsed + 1e-6 >= self.config.contact_debounce_seconds,
            raw,
            self.support,
        )
        sample["left_only_support"] = float(self.support[0] and not self.support[1])
        sample["right_only_support"] = float(self.support[1] and not self.support[0])
        sample["time_seconds"] = (len(self.samples) + 1) * self.timestep
        self.samples.append(sample)

    def summary(self):
        cfg = self.config
        samples = [s for s in self.samples if s["time_seconds"] > cfg.warmup_seconds]
        result = {"walking/accepted": 0, "walking/samples": len(samples)}
        if not samples:
            return result

        def mean(key):
            return float(np.mean([s[key] for s in samples]))

        velocity_rmse = math.sqrt(mean("velocity_error_sq"))
        yaw_rmse = math.sqrt(mean("yaw_rate_error_sq"))
        tilt = np.array([s["orientation_lateral_tilt"] for s in samples])
        tilt_rms = float(np.sqrt(np.mean(tilt**2)))
        tilt_peak = float(np.max(np.abs(tilt)))
        loaded = [s for s in samples if s["left_contact"] or s["right_contact"]]
        slip = (
            float(np.mean([s["stance_slip_speed"] for s in loaded])) if loaded else 0.0
        )
        sides = [
            1 if s["left_only_support"] else 2 if s["right_only_support"] else 0
            for s in samples
        ]
        supported_sides = [s for s in sides if s]
        alternations = sum(
            a != b for a, b in zip(supported_sides, supported_sides[1:], strict=False)
        )
        left = mean("left_only_support")
        right = mean("right_only_support")
        checks = {
            "duration": len(samples) * self.timestep + 1e-6
            >= cfg.minimum_window_seconds,
            "survival": not any(
                s["terminal"] or s["invalid_state"] for s in self.samples
            ),
            "velocity": velocity_rmse <= cfg.velocity_rmse_max,
            "yaw_rate": yaw_rmse <= cfg.yaw_rate_rmse_max,
            "stance_slip": bool(loaded) and slip <= cfg.stance_slip_max,
            "lateral_tilt_rms": tilt_rms <= cfg.lateral_tilt_rms_max,
            "lateral_tilt_peak": tilt_peak <= cfg.lateral_tilt_peak_max,
            "single_support": left + right >= cfg.single_support_min,
            "both_sides": min(left, right) >= cfg.each_side_support_min,
            "alternation": alternations >= cfg.minimum_support_alternations,
        }
        result.update(
            {f"walking/pass_{key}": int(value) for key, value in checks.items()}
        )
        result.update(
            {
                "walking/accepted": int(all(checks.values())),
                "walking/velocity_rmse": velocity_rmse,
                "walking/yaw_rate_rmse": yaw_rmse,
                "walking/stance_slip_speed": slip,
                "walking/lateral_tilt_rms": tilt_rms,
                "walking/lateral_tilt_peak": tilt_peak,
                "walking/lateral_tilt_peak_to_peak": float(np.ptp(tilt)),
                "walking/single_support": left + right,
                "walking/left_only_support": left,
                "walking/right_only_support": right,
                "walking/support_alternations": alternations,
            }
        )
        return result


def _lateral_tilt(up, yaw, forward_axis):
    """Signed tilt toward the command frame's left axis, in radians."""
    heading = yaw + math.atan2(forward_axis[1], forward_axis[0])
    lateral = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return math.atan2(float(np.dot(up, lateral)), float(up[2]))


def native_walking_sample(env, data, command, terminal):
    from bello_mujoco.evaluation.render import mujoco_foot_contact_and_normal_force

    model = env.model.mj_model
    frame = model.body(env._reward_frame_link("track_lin_vel_xy")).id
    rotation = data.xmat[frame].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    orientation_body = model.body(env._reward_link("flat_orientation")).id
    orientation_up = data.xmat[orientation_body].reshape(3, 3)[:, 2]
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        model.body(env._reward_link("track_lin_vel_xy")).id,
        velocity,
        0,
    )
    vx = math.cos(yaw) * velocity[3] + math.sin(yaw) * velocity[4]
    vy = -math.sin(yaw) * velocity[3] + math.cos(yaw) * velocity[4]
    angular = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        model.body(env._reward_link("track_ang_vel_z")).id,
        angular,
        0,
    )
    contact, force = mujoco_foot_contact_and_normal_force(env, data)
    terrain = set(map(int, env.model.terrain_geom_ids))
    foot_geoms = set(map(int, env.model.foot_geom_ids))
    load, weighted_slip = 0.0, 0.0
    for index in range(data.ncon):
        c = data.contact[index]
        other = (
            int(c.geom2)
            if int(c.geom1) in terrain
            else int(c.geom1)
            if int(c.geom2) in terrain
            else -1
        )
        if other not in foot_geoms:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        normal = abs(wrench[0])
        if normal <= env.config.contact.foot_contact_force_threshold:
            continue
        body = int(model.geom_bodyid[other])
        twist = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, twist, 0)
        point_velocity = twist[3:] + np.cross(twist[:3], c.pos - data.xpos[body])
        normal_axis = c.frame[:3]
        tangent = point_velocity - np.dot(point_velocity, normal_axis) * normal_axis
        weighted_slip += normal * np.linalg.norm(tangent)
        load += normal
    geometry = env._sole_reward_geometry
    ids = env.model.foot_body_ids
    centers = data.xpos[ids] + np.einsum(
        "nij,nj->ni", data.xmat[ids].reshape(-1, 3, 3), geometry.centers
    )
    result = {
        "x": data.xpos[frame, 0],
        "y": data.xpos[frame, 1],
        "yaw": yaw,
        "orientation_lateral_tilt": _lateral_tilt(
            orientation_up, yaw, env.model.robot.locomotion_forward_axis
        ),
        "vx": vx,
        "vy": vy,
        "yaw_rate": angular[2],
        "command_vx": command[0],
        "command_vy": command[1],
        "command_yaw_rate": command[2],
        "velocity_error_sq": (vx - command[0]) ** 2 + (vy - command[1]) ** 2,
        "yaw_rate_error_sq": (angular[2] - command[2]) ** 2,
        "stance_slip_speed": weighted_slip / load if load else 0.0,
        "terminal": float(terminal),
    }
    for i, side in enumerate(("left", "right")):
        result[f"{side}_contact"] = float(contact[i])
        result[f"{side}_normal_force"] = float(force[i])
        for axis, value in zip("xyz", centers[i], strict=True):
            result[f"{side}_sole_{axis}"] = float(value)
    return result
