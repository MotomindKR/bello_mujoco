//! Joint-aware target limiting. Hardwared independently rejects unsafe commands.
use anyhow::{Result, ensure};
use glam::DVec2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
struct Settings {
    motor_limits: Vec<[f64; 2]>,
    joint_limits: Vec<[f64; 2]>,
    joint_from_motor: Vec<Vec<f64>>,
    position_step_limits: Vec<f64>,
}

#[derive(Clone)]
struct Constraint {
    indices: [usize; 2],
    coefficients: DVec2,
    bounds: [f64; 2],
}

impl Constraint {
    fn value(&self, values: &[f64]) -> f64 {
        self.coefficients
            .dot(DVec2::new(values[self.indices[0]], values[self.indices[1]]))
    }
}

#[derive(Clone)]
pub struct TargetLimits {
    rows: Vec<Constraint>,
    groups: Vec<Vec<usize>>,
    default: Vec<f64>,
    steps: Vec<f64>,
}

#[derive(Serialize)]
pub struct LimitedTarget {
    pub requested_motor_target: Vec<f64>,
    pub motor_target: Vec<f64>,
    pub limited: bool,
}

impl TargetLimits {
    /// Convex transition envelope; both inputs must already include hardware bounds.
    pub fn transition_envelope(&self, other: &Self) -> Result<Self> {
        ensure!(
            self.rows.len() == other.rows.len()
                && self.groups == other.groups
                && self
                    .rows
                    .iter()
                    .zip(&other.rows)
                    .all(|(a, b)| a.indices == b.indices && a.coefficients == b.coefficients),
            "transition limit coordinate mappings differ"
        );
        let mut result = self.clone();
        for (row, other) in result.rows.iter_mut().zip(&other.rows) {
            row.bounds = [
                row.bounds[0].min(other.bounds[0]),
                row.bounds[1].max(other.bounds[1]),
            ];
        }
        for (step, other) in result.steps.iter_mut().zip(&other.steps) {
            *step = step.min(*other);
        }
        result.validate(&result.default)?;
        Ok(result)
    }

    /// Feedback allowance never widens outgoing commands or their first step.
    pub fn startup_target(&self, measured: &[f64], extra_margin: f64) -> Result<Vec<f64>> {
        ensure!(
            extra_margin.is_finite() && extra_margin >= 0.,
            "invalid startup feedback margin"
        );
        ensure!(
            measured.len() == self.default.len() && measured.iter().all(|v| v.is_finite()),
            "nonfinite or malformed startup feedback"
        );
        // Retain the encoding allowance; the configured margin is additional.
        let motor_tolerance =
            2. * robstride_protocol::CONTROL_POSITION_SPAN / u16::MAX as f64 + extra_margin;
        for (i, row) in self.rows.iter().enumerate() {
            let tolerance = row.coefficients.abs().element_sum() * motor_tolerance;
            let value = row.value(measured);
            ensure!(
                value >= row.bounds[0] - tolerance && value <= row.bounds[1] + tolerance,
                "startup feedback constraint {i}: {value} outside {:?} (feedback allowance {tolerance})",
                row.bounds
            );
        }
        let target = self.project(measured)?;
        for (i, ((next, previous), step)) in
            target.iter().zip(measured).zip(&self.steps).enumerate()
        {
            ensure!(
                (next - previous).abs() <= *step,
                "startup target {i} exceeds maximum position step from measured feedback"
            );
        }
        Ok(target)
    }

    pub fn new(document: &serde_json::Value, default: &[f32]) -> Result<Self> {
        let s: Settings = serde_json::from_value(document.clone())?;
        let n = default.len();
        ensure!(
            s.motor_limits.len() == n
                && s.joint_limits.len() == n
                && s.joint_from_motor.len() == n
                && s.position_step_limits.len() == n,
            "target limit shape mismatch"
        );
        ensure!(
            s.position_step_limits
                .iter()
                .all(|v| v.is_finite() && *v > 0.),
            "invalid position-step limit"
        );
        let mut connected = vec![vec![false; n]; n];
        let mut rows = Vec::new();
        for (i, &bounds) in s.motor_limits.iter().enumerate() {
            connected[i][i] = true;
            rows.push(Constraint {
                indices: [i, i],
                coefficients: DVec2::X,
                bounds,
            });
        }
        for (row, &bounds) in s.joint_from_motor.iter().zip(&s.joint_limits) {
            ensure!(
                row.len() == n && row.iter().all(|v| v.is_finite()),
                "invalid motor-to-joint mapping"
            );
            let ids: Vec<_> = row
                .iter()
                .enumerate()
                .filter_map(|(i, v)| (*v != 0.).then_some(i))
                .collect();
            ensure!(
                (1..=2).contains(&ids.len()),
                "target limiting requires direct or differential joints"
            );
            let a = ids[0];
            let b = *ids.last().unwrap();
            connected[a][b] = true;
            connected[b][a] = true;
            rows.push(Constraint {
                indices: [a, b],
                coefficients: DVec2::new(row[a], if a == b { 0. } else { row[b] }),
                bounds,
            });
        }
        ensure!(
            rows.iter()
                .all(|r| r.bounds.iter().all(|v| v.is_finite()) && r.bounds[0] < r.bounds[1]),
            "invalid position bounds"
        );
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    connected[i][j] |= connected[i][k] && connected[k][j];
                }
            }
        }
        let groups = (0..n)
            .filter(|&i| !(0..i).any(|j| connected[i][j]))
            .map(|i| (0..n).filter(|&j| connected[i][j]).collect())
            .collect();
        let limits = Self {
            rows,
            groups,
            default: default.iter().map(|v| *v as f64).collect(),
            steps: s.position_step_limits,
        };
        limits.validate(&limits.default)?;
        Ok(limits)
    }

    pub fn restrict_motor(&mut self, index: usize, lower: f64, upper: f64) -> Result<()> {
        ensure!(
            lower.is_finite() && upper.is_finite() && lower < upper,
            "invalid hardware motor bounds"
        );
        self.rows[index].bounds[0] = self.rows[index].bounds[0].max(lower);
        self.rows[index].bounds[1] = self.rows[index].bounds[1].min(upper);
        ensure!(
            self.rows[index].bounds[0] < self.rows[index].bounds[1],
            "hardware/policy motor limits do not overlap"
        );
        self.validate(&self.default)
    }

    pub fn validate(&self, values: &[f64]) -> Result<()> {
        ensure!(
            values.len() == self.default.len() && values.iter().all(|v| v.is_finite()),
            "nonfinite or malformed motor target"
        );
        for (i, row) in self.rows.iter().enumerate() {
            let value = row.value(values);
            ensure!(
                (row.bounds[0]..=row.bounds[1]).contains(&value),
                "target constraint {i}: {value} outside {:?}",
                row.bounds
            );
        }
        Ok(())
    }

    pub fn project(&self, target: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            target.len() == self.default.len() && target.iter().all(|v| v.is_finite()),
            "nonfinite or malformed requested motor target"
        );
        let mut result = target.to_vec();
        for group in &self.groups {
            let mut scale: f64 = 1.;
            for row in self.rows.iter().filter(|r| group.contains(&r.indices[0])) {
                let start = row.value(&self.default);
                let change = row.value(target) - start;
                if change > 0. {
                    scale = scale.min((row.bounds[1] - start) / change);
                }
                if change < 0. {
                    scale = scale.min((row.bounds[0] - start) / change);
                }
            }
            if scale < 1. {
                scale = scale.clamp(0., 1.) * (1. - 1e-6);
                for &i in group {
                    result[i] = self.default[i] + scale * (target[i] - self.default[i]);
                }
            }
        }
        self.validate(&result)?;
        Ok(result)
    }

    /// Minimum number of control steps along a feasible straight-line segment.
    /// One shared interpolation rate keeps joints synchronized without target lag.
    pub fn transition_steps(&self, start: &[f64], end: &[f64]) -> Result<f64> {
        self.validate(start)?;
        self.validate(end)?;
        Ok(start
            .iter()
            .zip(end)
            .zip(&self.steps)
            .map(|((a, b), step)| (b - a).abs() / step)
            .fold(0., f64::max))
    }

    pub fn constrain(&self, target: &[f64], previous: &[f64]) -> Result<LimitedTarget> {
        self.validate(previous)?;
        let mut result = self.project(target)?;
        for group in &self.groups {
            let mut scale: f64 = 1.;
            for &i in group {
                let delta = (result[i] - previous[i]).abs();
                if delta > self.steps[i] {
                    scale = scale.min(self.steps[i] / delta);
                }
            }
            if scale < 1. {
                scale *= 1. - 1e-6;
                for &i in group {
                    result[i] = previous[i] + scale * (result[i] - previous[i]);
                }
            }
        }
        self.validate(&result)?;
        Ok(LimitedTarget {
            requested_motor_target: target.to_vec(),
            limited: target != result,
            motor_target: result,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> TargetLimits {
        TargetLimits::new(
            &serde_json::json!({
                "motor_limits": [[-2., 2.], [-2., 2.], [-3., 3.]],
                "joint_limits": [[-0.4, 0.6], [-0.3, 0.5], [-1., 2.]],
                "joint_from_motor": [[0.5, 0.5, 0.], [-0.5, 0.5, 0.], [0., 0., 1.]],
                "position_step_limits": [0.1, 0.2, 0.3]
            }),
            &[0., 0., 0.],
        )
        .unwrap()
    }

    #[test]
    fn coupled_envelope_and_steps_are_preserved() {
        let limits = limits();
        let projected = limits.project(&[1.5, 1.5, 0.2]).unwrap();
        assert!((projected[0] - 0.6 * (1. - 1e-6)).abs() < 1e-12);
        assert_eq!(projected[0], projected[1]);
        assert_eq!(projected[2], 0.2);
        let result = limits.constrain(&[1.5, 1.5, 0.2], &[0.; 3]).unwrap();
        assert!(result.limited);
        assert_eq!(result.requested_motor_target, [1.5, 1.5, 0.2]);
        assert!((result.motor_target[0] - 0.1 * (1. - 1e-6)).abs() < 1e-12);
        assert_eq!(result.motor_target[0], result.motor_target[1]);
        assert_eq!(result.motor_target[2], 0.2);
        assert_eq!(limits.project(&projected).unwrap(), projected);
    }

    #[test]
    fn repeated_requests_remain_feasible() {
        let limits = limits();
        let mut previous = vec![0.; 3];
        for i in 0..1000 {
            let target = [
                (i as f64).sin() * 4.,
                (i as f64 * 0.3).cos() * 4.,
                (i as f64 * 0.1).sin() * 4.,
            ];
            let result = limits.constrain(&target, &previous).unwrap();
            limits.validate(&result.motor_target).unwrap();
            for ((next, prev), step) in result.motor_target.iter().zip(&previous).zip(&limits.steps)
            {
                assert!((next - prev).abs() <= *step);
            }
            previous = result.motor_target;
        }
    }

    #[test]
    fn bad_inputs_fail_and_live_bounds_can_only_tighten() {
        let mut limits = limits();
        assert!(limits.project(&[f64::NAN, 0., 0.]).is_err());
        assert!(limits.constrain(&[0.; 3], &[2., 2., 0.]).is_err());
        assert!(limits.project(&[]).is_err());
        assert!(limits.restrict_motor(0, f64::NAN, 1.).is_err());
        limits.restrict_motor(2, -0.1, 0.1).unwrap();
        assert!(limits.project(&[0., 0., 0.2]).unwrap()[2] < 0.1);
        limits.restrict_motor(2, -3., 3.).unwrap();
        assert!(limits.project(&[0., 0., 0.2]).unwrap()[2] < 0.1);
    }

    #[test]
    fn transition_reaches_wider_task_and_returns_without_violating_hardware_or_steps() {
        let standing = TargetLimits::new(
            &serde_json::json!({
                "motor_limits": [[0., 0.8]], "joint_limits": [[0., 2.5]],
                "joint_from_motor": [[1.]], "position_step_limits": [0.1]
            }),
            &[0.4],
        )
        .unwrap();
        let mut motion = TargetLimits::new(
            &serde_json::json!({
                "motor_limits": [[0., 2.5]], "joint_limits": [[0., 2.5]],
                "joint_from_motor": [[1.]], "position_step_limits": [0.2]
            }),
            &[0.4],
        )
        .unwrap();
        motion.restrict_motor(0, 0., 1.8).unwrap();
        let transition = standing.transition_envelope(&motion).unwrap();
        let mut previous = vec![0.4];
        for _ in 0..30 {
            let next = transition
                .constrain(&[2.4], &previous)
                .unwrap()
                .motor_target;
            assert!(next[0] <= 1.8 && (next[0] - previous[0]).abs() <= 0.1);
            previous = next;
        }
        assert!(previous[0] > 1.7);
        assert!(standing.validate(&previous).is_err());
        motion.validate(&previous).unwrap();
        // A completed timed blend does not imply we may switch to standing yet.
        previous = transition
            .constrain(&[0.4], &previous)
            .unwrap()
            .motor_target;
        assert!(standing.validate(&previous).is_err());
        for _ in 0..30 {
            let next = transition
                .constrain(&[0.4], &previous)
                .unwrap()
                .motor_target;
            assert!((next[0] - previous[0]).abs() <= 0.1);
            previous = next;
        }
        standing.validate(&previous).unwrap();
        assert!(standing.constrain(&[2.4], &previous).unwrap().motor_target[0] <= 0.8);
    }

    #[test]
    fn transition_preserves_coupled_rows_and_rejects_other_mappings() {
        let standing = limits();
        let mut motion = standing.clone();
        motion.rows[3].bounds = [-0.8, 1.2];
        motion.restrict_motor(0, -0.7, 0.7).unwrap();
        let mut standing = standing;
        standing.restrict_motor(0, -0.7, 0.7).unwrap();
        let transition = standing.transition_envelope(&motion).unwrap();
        let mut previous = vec![0.; 3];
        for _ in 0..30 {
            previous = transition
                .constrain(&[2., 2., 0.], &previous)
                .unwrap()
                .motor_target;
        }
        assert!(previous[0] <= 0.7 && previous[0] > 0.6);
        assert_eq!(previous[0], previous[1]);
        motion.validate(&previous).unwrap();
        motion.rows[3].coefficients = DVec2::new(0.5, -0.5);
        assert!(standing.transition_envelope(&motion).is_err());
    }

    #[test]
    fn startup_quantization_never_widens_command_bounds() {
        let limits = TargetLimits::new(
            &serde_json::json!({
                "motor_limits": [[0., 1.2]], "joint_limits": [[0., 2.]],
                "joint_from_motor": [[1.]], "position_step_limits": [0.1]
            }),
            &[0.45],
        )
        .unwrap();
        for measured in [-0.00019180590524214836, 1.200191805905242] {
            assert!(limits.validate(&[measured]).is_err());
            let target = limits.startup_target(&[measured], 0.).unwrap();
            limits.validate(&target).unwrap();
            assert!((target[0] - measured).abs() < 0.001);
        }
        assert_eq!(limits.startup_target(&[0.3], 0.).unwrap(), [0.3]);
        for measured in [-0.002, 1.202, f64::NAN, f64::INFINITY] {
            assert!(limits.startup_target(&[measured], 0.).is_err());
        }
        assert!(limits.startup_target(&[], 0.).is_err());
        let target = limits.startup_target(&[1.202], 0.002).unwrap();
        limits.validate(&target).unwrap();
        assert!(target[0] <= 1.2);
        assert!(limits.startup_target(&[1.204], 0.002).is_err());
        for margin in [-0.001, f64::NAN, f64::INFINITY] {
            assert!(limits.startup_target(&[0.3], margin).is_err());
        }
    }

    #[test]
    fn startup_propagates_quantization_through_coupling_and_checks_first_step() {
        let limits = TargetLimits::new(
            &serde_json::json!({
                "motor_limits": [[-2., 2.], [-2., 2.]],
                "joint_limits": [[-1.2, 1.2], [-0.3, 0.5]],
                "joint_from_motor": [[2., 2.], [-0.5, 0.5]],
                "position_step_limits": [0.1, 0.1]
            }),
            &[0., 0.],
        )
        .unwrap();
        let measured = [0.3006, 0.3006];
        assert!(limits.validate(&measured).is_err());
        limits
            .validate(&limits.startup_target(&measured, 0.).unwrap())
            .unwrap();
        assert!(limits.startup_target(&[0.302, 0.302], 0.).is_err());
        limits
            .validate(&limits.startup_target(&[0.302, 0.302], 0.002).unwrap())
            .unwrap();
        let tight_steps = TargetLimits {
            steps: vec![0.0001; 2],
            ..limits
        };
        assert!(tight_steps.startup_target(&measured, 0.).is_err());
        assert!(tight_steps.startup_target(&[0.302, 0.302], 0.002).is_err());
    }
}
