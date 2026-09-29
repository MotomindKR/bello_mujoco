//! Assisted, operator-paced startup. No policy inference runs during this sequence.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LyingConfig {
    pub hip_pitch_radians: f64,
    pub knee_radians: f64,
    pub straighten_speed_rps: f64,
    pub ankle_engage_seconds: f64,
}

pub struct LyingPose {
    bent: Vec<f64>,
    ankles: Vec<usize>,
    duration: f64,
    engage_seconds: f64,
}

#[derive(Deserialize)]
struct Mapping {
    joint_names: Vec<String>,
    default_joint_position: Vec<f64>,
    motor_from_joint: Vec<Vec<f64>>,
    joint_from_motor: Vec<Vec<f64>>,
}

impl LyingPose {
    pub fn new(
        config: &LyingConfig,
        policy: &crate::Policy,
        limits: &crate::target_limits::TargetLimits,
    ) -> Result<Self> {
        ensure!(
            config.hip_pitch_radians.is_finite() && config.knee_radians.is_finite(),
            "invalid lying start angles"
        );
        ensure!(
            config.straighten_speed_rps.is_finite()
                && config.straighten_speed_rps > 0.
                && config.straighten_speed_rps <= 0.5,
            "lying start speed must be in (0, 0.5] rad/s"
        );
        ensure!(
            config.ankle_engage_seconds.is_finite()
                && (1. ..=60.).contains(&config.ankle_engage_seconds),
            "ankle engagement must take 1 to 60 seconds"
        );
        let m: Mapping = serde_json::from_value(policy.document.clone())?;
        let n = policy.metadata.coordinates.len();
        ensure!(
            m.joint_names.len() == n
                && m.joint_names
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == n
                && m.default_joint_position.len() == n
                && m.default_joint_position.iter().all(|v| v.is_finite()),
            "invalid startup joint mapping"
        );
        for matrix in [&m.motor_from_joint, &m.joint_from_motor] {
            ensure!(
                matrix.len() == n
                    && matrix
                        .iter()
                        .all(|r| r.len() == n && r.iter().all(|v| v.is_finite())),
                "invalid startup coordinate mapping"
            );
        }
        for i in 0..n {
            for j in 0..n {
                let product: f64 = (0..n)
                    .map(|k| m.motor_from_joint[i][k] * m.joint_from_motor[k][j])
                    .sum();
                ensure!(
                    (product - if i == j { 1. } else { 0. }).abs() < 1e-8,
                    "startup mappings are not inverses"
                );
            }
        }
        for (row, nominal) in m
            .motor_from_joint
            .iter()
            .zip(&policy.metadata.default_position)
        {
            let mapped: f64 = row
                .iter()
                .zip(&m.default_joint_position)
                .map(|(a, b)| a * b)
                .sum();
            ensure!(
                (mapped - f64::from(*nominal)).abs() < 1e-5,
                "startup nominal joint/motor poses disagree"
            );
        }
        let index = |name: &str| {
            m.joint_names
                .iter()
                .position(|n| n == name)
                .with_context(|| format!("lying start requires {name}"))
        };
        let mut joints = m.default_joint_position.clone();
        let mut travel: f64 = 0.;
        for side in ["left", "right"] {
            for (joint, value) in [
                ("hip_pitch", config.hip_pitch_radians),
                ("knee", config.knee_radians),
            ] {
                let i = index(&format!("{side}_{joint}_joint"))?;
                travel = travel.max((value - joints[i]).abs());
                ensure!(
                    if joint == "hip_pitch" {
                        value < joints[i]
                    } else {
                        value > joints[i]
                    },
                    "lying start must bend hips and knees beyond nominal"
                );
                joints[i] = value;
            }
        }
        let ankle_joints = [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        .map(index)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        let ankles: Vec<_> = (0..n)
            .filter(|&i| {
                ankle_joints
                    .iter()
                    .any(|&j| m.motor_from_joint[i][j].abs() > 1e-12)
            })
            .collect();
        ensure!(!ankles.is_empty(), "missing startup ankle actuators");
        for &i in &ankles {
            ensure!(
                (0..n).all(|j| ankle_joints.contains(&j) || m.motor_from_joint[i][j].abs() < 1e-12),
                "ankle actuator also drives a non-ankle joint"
            );
        }
        let bent = m
            .motor_from_joint
            .iter()
            .map(|row| row.iter().zip(&joints).map(|(a, b)| a * b).sum())
            .collect::<Vec<_>>();
        limits
            .validate(&bent)
            .context("lying start pose exceeds target limits")?;
        let nominal: Vec<f64> = policy
            .metadata
            .default_position
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let duration = (travel / config.straighten_speed_rps).max(
            limits.transition_steps(&bent, &nominal)?
                * policy.metadata.control_timestep
                * (1. + 1e-6),
        );
        ensure!(duration.is_finite(), "invalid straightening duration");
        Ok(Self {
            bent,
            ankles,
            duration,
            engage_seconds: config.ankle_engage_seconds,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Ramping,
    Bending,
    AwaitingLift,
    Straightening,
    Upright,
    EngagingAnkles,
    Holding,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ramping => "ramping",
            Self::Bending => "bending",
            Self::AwaitingLift => "awaiting lift",
            Self::Straightening => "straightening",
            Self::Upright => "upright",
            Self::EngagingAnkles => "engaging ankles",
            Self::Holding => "holding",
        }
    }
}

pub struct Sequence<'a> {
    lying: Option<&'a LyingPose>,
    initial: Vec<f64>,
    nominal: Vec<f64>,
    pub target: Vec<f64>,
    pub gains: Vec<[f32; 2]>,
    standing_gains: Vec<[f32; 2]>,
    phase: Phase,
    pub progress: f64,
    ramp_seconds: f64,
}

impl<'a> Sequence<'a> {
    pub fn new(
        lying: Option<&'a LyingPose>,
        initial: Vec<f64>,
        nominal: Vec<f64>,
        gains: Vec<[f32; 2]>,
        ramp_seconds: f64,
    ) -> Self {
        let mut result = Self {
            lying,
            target: initial.clone(),
            initial,
            nominal,
            gains: gains.clone(),
            standing_gains: gains,
            phase: if lying.is_some() {
                Phase::Bending
            } else {
                Phase::Ramping
            },
            progress: 0.,
            ramp_seconds,
        };
        result.free_ankles();
        result
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    fn free_ankles(&mut self) {
        if let Some(pose) = self.lying {
            for &i in &pose.ankles {
                self.gains[i] = [0., 0.];
            }
        }
    }

    pub fn requires_upright(&self) -> bool {
        !matches!(self.phase, Phase::Bending | Phase::AwaitingLift)
    }

    pub fn can_confirm(&self, upright: bool) -> bool {
        upright
            && matches!(
                self.phase,
                Phase::AwaitingLift | Phase::Upright | Phase::Holding
            )
    }

    /// Returns true only after the final confirmation and ankle engagement.
    pub fn tick(
        &mut self,
        dt: f64,
        connected: bool,
        confirm: bool,
        forward: f32,
        measured: &[f64],
        applied: &[f64],
    ) -> bool {
        if !connected {
            return false;
        }
        let reached = self
            .target
            .iter()
            .zip(applied)
            .all(|(a, b)| (a - b).abs() < 1e-7);
        match self.phase {
            Phase::Bending | Phase::Ramping => {
                if self.progress == 1. && reached {
                    self.phase = if self.lying.is_some() {
                        Phase::AwaitingLift
                    } else {
                        Phase::Holding
                    };
                    return false;
                }
                self.progress = (self.progress + dt / self.ramp_seconds).min(1.);
                let end = self.lying.map_or(&self.nominal, |p| &p.bent);
                self.target = blend(&self.initial, end, self.progress);
            }
            Phase::AwaitingLift if confirm => {
                self.phase = Phase::Straightening;
                self.progress = 0.;
            }
            Phase::Straightening => {
                if self.progress == 1. && reached {
                    self.phase = Phase::Upright;
                    return false;
                }
                let pose = self.lying.unwrap();
                self.progress =
                    (self.progress + f64::from(forward.clamp(0., 1.)) * dt / pose.duration).min(1.);
                self.target = blend(&pose.bent, &self.nominal, self.progress);
            }
            Phase::Upright if confirm => {
                self.phase = Phase::EngagingAnkles;
                self.progress = 0.;
                self.initial = self.target.clone();
                for &i in &self.lying.unwrap().ankles {
                    self.initial[i] = measured[i];
                }
                self.target = self.initial.clone();
            }
            Phase::EngagingAnkles => {
                if self.progress == 0. && !reached {
                    return false;
                }
                if self.progress == 1. && reached {
                    return true;
                }
                self.progress = (self.progress + dt / self.lying.unwrap().engage_seconds).min(1.);
                self.target = blend(&self.initial, &self.nominal, self.progress);
                for &i in &self.lying.unwrap().ankles {
                    self.gains[i] = self.standing_gains[i].map(|g| g * self.progress as f32);
                }
            }
            Phase::Holding if confirm => return true,
            Phase::AwaitingLift | Phase::Upright | Phase::Holding => {}
        }
        false
    }
}

fn blend(start: &[f64], end: &[f64], t: f64) -> Vec<f64> {
    start
        .iter()
        .zip(end)
        .map(|(a, b)| a + t * (b - a))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assisted_lift_requires_confirmation_pauses_and_restores_ankles_gradually() {
        let pose = LyingPose {
            bent: vec![-1., 2., 0.],
            ankles: vec![2],
            duration: 10.,
            engage_seconds: 2.,
        };
        let mut sequence = Sequence::new(
            Some(&pose),
            vec![0.; 3],
            vec![0.; 3],
            vec![[10., 1.]; 3],
            1.,
        );
        assert_eq!(sequence.gains[2], [0., 0.]);
        assert!(!sequence.requires_upright());
        let tick = |s: &mut Sequence<'_>, dt, connected, confirm, forward| {
            let applied = s.target.clone();
            s.tick(dt, connected, confirm, forward, &[0., 0., 0.4], &applied)
        };
        assert!(!tick(&mut sequence, 1., true, false, 0.));
        assert!(!tick(&mut sequence, 0.02, true, false, 0.));
        assert_eq!(sequence.phase, Phase::AwaitingLift);
        assert!(!sequence.can_confirm(false));
        assert!(sequence.can_confirm(true));
        tick(&mut sequence, 2., true, false, 1.);
        assert_eq!(sequence.target, pose.bent);
        tick(&mut sequence, 0.02, true, true, 0.);
        assert_eq!(sequence.phase, Phase::Straightening);
        assert!(sequence.requires_upright());
        tick(&mut sequence, 1., true, false, 0.5);
        assert!((sequence.progress - 0.05).abs() < 1e-9);
        assert!((sequence.target[1] - 1.9).abs() < 1e-9);
        let held = sequence.target.clone();
        tick(&mut sequence, 1., true, true, 0.);
        tick(&mut sequence, 1., true, false, -1.);
        tick(&mut sequence, 1., false, true, 1.);
        assert_eq!(sequence.target, held);
        for _ in 0..100 {
            tick(&mut sequence, 0.1, true, false, 1.);
        }
        assert_eq!(sequence.phase, Phase::Upright);
        assert_eq!(sequence.gains[2], [0., 0.]);
        assert!(!tick(&mut sequence, 1., true, false, 0.));
        assert!(!tick(&mut sequence, 0.02, true, true, 0.));
        assert_eq!(sequence.phase, Phase::EngagingAnkles);
        tick(&mut sequence, 1., true, false, 0.);
        assert_eq!(sequence.gains[2], [5., 0.5]);
        assert!((sequence.target[2] - 0.2).abs() < 1e-9);
        tick(&mut sequence, 1., false, false, 0.);
        assert_eq!(sequence.gains[2], [5., 0.5]);
        tick(&mut sequence, 1., true, false, 0.);
        assert!(tick(&mut sequence, 0.02, true, false, 0.));
        assert_eq!(sequence.gains[2], [10., 1.]);
    }

    #[test]
    fn crane_path_keeps_nominal_hold_and_separate_stand_confirmation() {
        let mut sequence = Sequence::new(None, vec![0.], vec![1.], vec![[10., 1.]], 1.);
        assert!(sequence.requires_upright());
        assert!(!sequence.tick(1., true, true, 1., &[0.], &[0.]));
        assert_eq!(sequence.phase, Phase::Ramping);
        assert!(!sequence.tick(0.02, true, false, 0., &[1.], &[1.]));
        assert_eq!(sequence.phase, Phase::Holding);
        assert!(!sequence.tick(1., false, true, 0., &[1.], &[1.]));
        assert!(sequence.tick(0.02, true, true, 0., &[1.], &[1.]));
    }
}
