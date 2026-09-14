"""Reward-independent scoring of fixed zero-command diagnostic rollouts."""

from dataclasses import dataclass, field
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate_standing_frame(frame, sample, push_at_seconds):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    phase = (
        "quiet"
        if push_at_seconds is None
        else ("pre-push" if sample["time_seconds"] < push_at_seconds else "post-push")
    )
    text = (
        f"{phase} t={sample['time_seconds']:.2f}s  settled={int(sample['settled'])}\n"
        f"speed={sample['speed']:.2f}m/s yaw={sample['yaw_rate']:.2f}rad/s "
        f"contact L/R={int(sample['left_contact'])}/{int(sample['right_contact'])}\n"
        f"stagger={sample['foreaft_error'] * 100:.1f}cm width error={sample['width_error'] * 100:.1f}cm\n"
        f"torque sat={sample['torque_saturation']:.0%}"
    )
    font = ImageFont.load_default(size=14)
    bounds = draw.multiline_textbbox((6, 6), text, font=font)
    draw.rectangle((0, 0, image.width, bounds[3] + 6), fill="black")
    draw.multiline_text((6, 6), text, fill="white", font=font)
    return np.asarray(image)


@dataclass
class SettlingDiagnostics:
    timestep: float
    push_at_step: int | None = None
    samples: list[dict[str, float]] = field(default_factory=list)
    support: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))
    contact_elapsed: np.ndarray = field(default_factory=lambda: np.zeros(2))
    settled_duration: float = 0.0

    def record(self, sample: dict[str, float]) -> None:
        sample = dict(sample)
        if not all(math.isfinite(value) for value in sample.values()):
            sample = {
                key: value if math.isfinite(value) else 0.0
                for key, value in sample.items()
            }
            sample["terminal"] = 1.0
            sample["invalid_state"] = 1.0
        else:
            sample["invalid_state"] = 0.0
        raw = np.array([sample["left_contact"], sample["right_contact"]], dtype=bool)
        changed = raw != self.support
        self.contact_elapsed = np.where(
            changed, self.contact_elapsed + self.timestep, 0
        )
        self.support = np.where(self.contact_elapsed + 1e-6 >= 0.04, raw, self.support)
        step = len(self.samples)
        if step == self.push_at_step:
            self.settled_duration = 0.0
        stable = (
            all(math.isfinite(value) for value in sample.values())
            and bool(self.support.all())
            and sample["speed"] < 0.05
            and abs(sample["yaw_rate"]) < 0.1
            and sample["tilt"] < math.radians(10)
            and sample["foreaft_error"] < 0.03
            and sample["width_error"] < 0.03
            and sample["heading_error"] < math.radians(10)
            and not sample["terminal"]
        )
        self.settled_duration = self.settled_duration + self.timestep if stable else 0.0
        sample["settled"] = float(self.settled_duration >= 1.0 - 1e-6)
        sample["double_support"] = float(self.support.all())
        sample["left_only_support"] = float(self.support[0] and not self.support[1])
        sample["right_only_support"] = float(self.support[1] and not self.support[0])
        sample["time_seconds"] = (step + 1) * self.timestep
        self.samples.append(sample)

    def summary(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        segments = {"standing": self.samples}
        if self.push_at_step is not None:
            segments["pre_push"] = self.samples[: self.push_at_step]
            segments["post_push"] = self.samples[self.push_at_step :]
        for prefix, samples in segments.items():
            result[f"{prefix}/samples"] = len(samples)
            if not samples:
                result[f"{prefix}/settled_at_end"] = 0
                result[f"{prefix}/time_to_settle_seconds"] = -1.0
                continue
            for name in (
                "double_support",
                "left_only_support",
                "right_only_support",
                "foreaft_error",
                "width_error",
                "heading_error",
                "tilt",
                "torque_saturation",
                "self_collision_contacts",
            ):
                result[f"{prefix}/mean_{name}"] = float(
                    np.mean([s[name] for s in samples])
                )
            result[f"{prefix}/speed_rms"] = float(
                np.sqrt(np.mean([s["speed"] ** 2 for s in samples]))
            )
            result[f"{prefix}/yaw_rate_rms"] = float(
                np.sqrt(np.mean([s["yaw_rate"] ** 2 for s in samples]))
            )
            start = samples[0]
            result[f"{prefix}/max_position_drift_m"] = max(
                math.hypot(s["x"] - start["x"], s["y"] - start["y"]) for s in samples
            )
            result[f"{prefix}/max_yaw_drift_radians"] = max(
                abs(
                    math.atan2(
                        math.sin(s["yaw"] - start["yaw"]),
                        math.cos(s["yaw"] - start["yaw"]),
                    )
                )
                for s in samples
            )
            settled = [i for i, s in enumerate(samples) if s["settled"]]
            result[f"{prefix}/time_to_settle_seconds"] = (
                (settled[0] + 1) * self.timestep if settled else -1.0
            )
            result[f"{prefix}/settled_at_end"] = int(samples[-1]["settled"])
        if self.push_at_step is not None:
            result["push_precondition_settled"] = result["pre_push/settled_at_end"]
            result["push_recovery_settled"] = int(
                bool(result["pre_push/settled_at_end"])
                and bool(result["post_push/settled_at_end"])
                and not any(s["terminal"] for s in self.samples)
            )
        return result
