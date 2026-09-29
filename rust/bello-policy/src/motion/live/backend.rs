//! Observation-specific inference behind the shared task transition lifecycle.
use super::super::{Feedback, Motion, ObservationMode, PolicyFailure};
use crate::{Policy, locomotion, tracking};
use anyhow::Result;
use glam::DQuat;
use serde_json::{Value, json};

pub(super) enum Backend {
    Mimic(Box<Motion>),
    Sonic(Box<tracking::Tracking>),
}

pub(super) struct Inference {
    pub motor_target: Vec<f32>,
    pub raw_action: Vec<f32>,
    pub log: Value,
}
impl Backend {
    pub fn policy(&self) -> &Policy {
        match self {
            Self::Mimic(m) => &m.policy,
            Self::Sonic(t) => &t.sonic.policy,
        }
    }
    pub fn policy_mut(&mut self) -> &mut Policy {
        match self {
            Self::Mimic(m) => &mut m.policy,
            Self::Sonic(t) => &mut t.sonic.policy,
        }
    }
    pub fn reset(&mut self) {
        match self {
            Self::Mimic(m) => m.reset(),
            Self::Sonic(t) => t.sonic.reset(),
        }
    }
    pub fn duration(&self) -> f64 {
        match self {
            Self::Mimic(m) => m.duration(),
            Self::Sonic(_) => f64::INFINITY,
        }
    }
    pub fn observation_mode(&self) -> ObservationMode {
        match self {
            Self::Mimic(m) => m.settings.environment.observation_mode,
            Self::Sonic(_) => ObservationMode::Proprioceptive,
        }
    }
    pub fn root_from_fixed_frame(&mut self, frame: &str) -> Result<DQuat> {
        match self {
            Self::Mimic(m) => m.root_from_fixed_frame(frame),
            Self::Sonic(t) => Ok(t.sonic.root_from_imu()),
        }
    }
    pub fn require_root_pose_frame(&mut self, frame: &str) -> Result<()> {
        match self {
            Self::Mimic(m) => m.require_root_pose_frame(frame),
            Self::Sonic(_) => anyhow::bail!("SONIC uses IMU/joint feedback"),
        }
    }
    pub fn reference_orientation(&self, time: f64) -> DQuat {
        match self {
            Self::Mimic(m) => {
                let sample = m.clip.sample(time);
                super::super::rotation::from_wxyz(sample.qpos[3..7].try_into().unwrap())
            }
            Self::Sonic(t) => t.reference_orientation(),
        }
    }
    pub fn refresh(&mut self) {
        if let Self::Sonic(t) = self {
            t.refresh();
        }
    }
    pub fn tracking_status(&self) -> Option<&tracking::Status> {
        match self {
            Self::Sonic(t) => Some(t.status()),
            _ => None,
        }
    }
    pub fn step(
        &mut self,
        feedback: &Feedback,
        hardware: &locomotion::Feedback,
    ) -> Result<Inference> {
        match self {
            Self::Mimic(m) => {
                let step = m.step(feedback)?;
                Ok(Inference {
                    log: json!(step),
                    motor_target: step.motor_target,
                    raw_action: step.raw_action,
                })
            }
            Self::Sonic(t) => {
                let step = t
                    .step(hardware, feedback.initialize)
                    .map_err(|e| e.context(PolicyFailure))?;
                Ok(Inference {
                    log: json!(step),
                    motor_target: step.motor_target,
                    raw_action: step.raw_action,
                })
            }
        }
    }
}
