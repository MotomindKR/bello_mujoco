//! Task phases only: lease, command acknowledgements, and shutdown stay in control.

use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use glam::DQuat;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Feedback, Motion, pose};
use crate::locomotion::{Feedback as HardwareFeedback, Locomotion};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub standing_policy: PathBuf,
    pub pose: pose::Config,
    pub start_time_seconds: f64,
    pub transition_seconds: f64,
    pub max_tilt_radians: f64,
    pub max_raw_action_abs: f32,
}

/// A first interrupt requests a blended return without cancelling an in-flight ACK.
/// A second interrupt, or an interrupt outside motion, takes the ordinary STOP path.
#[derive(Clone, Default)]
pub struct Interrupt(Arc<AtomicU8>);

impl Interrupt {
    pub fn request_return(&self) -> bool {
        self.0
            .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

struct Transition {
    initial: Vec<f32>,
    step: usize,
    steps: usize,
}

enum Phase {
    Standing,
    ToMotion(Transition),
    Motion(Instant),
    ToStanding(Transition),
}

pub struct Output {
    pub target: Vec<f32>,
    pub log: Value,
    pub period: Duration,
}

pub struct Live {
    motion: Motion,
    poses: pose::Source,
    config: Config,
    root_from_imu: DQuat,
    phase: Phase,
    pub interrupt: Interrupt,
}

impl Live {
    pub async fn connect(
        mut motion: Motion,
        standing: &Locomotion,
        config: Config,
    ) -> Result<Self> {
        let a = &motion.policy.metadata;
        let b = &standing.policy.metadata;
        ensure!(
            a.robot == b.robot && a.coordinates == b.coordinates && a.kp == b.kp && a.kd == b.kd,
            "standing/motion motor coordinates or gains differ"
        );
        for key in ["joint_names", "joint_from_motor"] {
            ensure!(
                motion.policy.document.get(key) == standing.policy.document.get(key),
                "standing/motion {key} differ"
            );
        }
        ensure!(
            motion.policy.document["hardware_imu_frame"].as_str() == Some(standing.imu_frame()),
            "standing/motion IMU frame mismatch"
        );
        ensure!(
            config.pose.tracked_frame == motion.root_frame()?,
            "pose must track the motion model's floating-base frame"
        );
        ensure!(
            config.start_time_seconds.is_finite()
                && config.start_time_seconds >= 0.
                && config.start_time_seconds < motion.duration(),
            "invalid motion start time"
        );
        for value in [
            config.transition_seconds,
            config.max_tilt_radians,
            f64::from(config.max_raw_action_abs),
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 86400.,
                "invalid motion control limit/timing"
            );
        }
        let observation_size = a.observation_size;
        let root_from_imu = motion.root_from_fixed_frame(standing.imu_frame())?;
        // Warm up the graph before acquiring a lease, not on the first motion tick.
        motion.policy.infer(&vec![0.; observation_size])?;
        let poses = pose::Source::connect(config.pose.clone()).await?;
        Ok(Self {
            motion,
            poses,
            config,
            root_from_imu,
            phase: Phase::Standing,
            interrupt: Interrupt::default(),
        })
    }

    pub fn motion_period(&self) -> f64 {
        self.motion.policy.metadata.control_timestep
    }

    pub fn tilt_limit(&self, standing: f64) -> f64 {
        if matches!(self.phase, Phase::Standing) {
            standing
        } else {
            self.config.max_tilt_radians
        }
    }

    pub fn check_feedback(&self, feedback: &HardwareFeedback) -> Result<()> {
        self.pose(feedback).map(|_| ())
    }

    fn pose(&self, feedback: &HardwareFeedback) -> Result<pose::Pose> {
        let pose = self.poses.sample()?;
        let imu =
            super::rotation::checked_quaternion(DQuat::from_array(feedback.orientation_xyzw))?;
        ensure!(
            (pose.orientation * self.root_from_imu)
                .angle_between(imu)
                .abs()
                <= self.config.pose.max_imu_orientation_error,
            "pose/IMU orientation mismatch"
        );
        Ok(pose)
    }

    fn transition(&self, initial: &[f32], period: f64) -> Transition {
        Transition {
            initial: initial.to_vec(),
            step: 0,
            steps: (self.config.transition_seconds / period).ceil() as usize,
        }
    }

    pub fn step(
        &mut self,
        standing: &mut Locomotion,
        feedback: &HardwareFeedback,
        request_motion: bool,
        previous_target: &[f32],
    ) -> Result<Output> {
        let pose = self.pose(feedback)?;
        let standing_period = standing.policy.metadata.control_timestep;
        if request_motion {
            ensure!(
                matches!(self.phase, Phase::Standing),
                "MOTION is only accepted while standing"
            );
            self.phase = Phase::ToMotion(self.transition(previous_target, self.motion_period()));
            self.motion.reset();
            self.interrupt.0.store(1, Ordering::SeqCst);
            eprintln!("Transitioning to motion tracking.");
        }
        let now = Instant::now();
        let complete = matches!(self.phase, Phase::Motion(started)
            if self.config.start_time_seconds + now.saturating_duration_since(started).as_secs_f64() >= self.motion.duration());
        if complete || self.interrupt.0.load(Ordering::SeqCst) == 2 {
            self.phase = Phase::ToStanding(self.transition(previous_target, standing_period));
            self.interrupt.0.store(0, Ordering::SeqCst);
            standing.reset();
            eprintln!("Transitioning back to standing.");
        }
        let (target, policy_log, phase_name, reference_time, period) = match &self.phase {
            Phase::Standing | Phase::ToStanding(_) => {
                let step = standing.step(feedback)?;
                let name = if matches!(self.phase, Phase::Standing) {
                    "standing"
                } else {
                    "transition_to_standing"
                };
                (
                    step.motor_target.clone(),
                    json!(step),
                    name,
                    None,
                    standing_period,
                )
            }
            Phase::ToMotion(_) | Phase::Motion(_) => {
                let time = self.config.start_time_seconds
                    + match self.phase {
                        Phase::Motion(started) => {
                            now.saturating_duration_since(started).as_secs_f64()
                        }
                        _ => 0.,
                    };
                let initialize = matches!(&self.phase, Phase::ToMotion(t) if t.step == 0);
                let sample = Feedback {
                    motor_position: feedback.motor_position.clone(),
                    motor_velocity: feedback.motor_velocity.clone(),
                    position: pose.position.to_array(),
                    orientation_xyzw: pose.orientation.to_array(),
                    linear_velocity_world: pose.linear_velocity.to_array(),
                    angular_velocity_local: pose.angular_velocity.to_array(),
                    reference_time: time,
                    initialize,
                };
                let step = self.motion.step(&sample)?;
                ensure!(
                    step.raw_action
                        .iter()
                        .all(|a| a.abs() <= self.config.max_raw_action_abs),
                    "implausible motion action"
                );
                let name = if initialize || matches!(self.phase, Phase::ToMotion(_)) {
                    "transition_to_motion"
                } else {
                    "motion"
                };
                (
                    step.motor_target.clone(),
                    json!(step),
                    name,
                    Some(time),
                    self.motion_period(),
                )
            }
        };
        let mut target = target;
        if let Phase::ToMotion(t) | Phase::ToStanding(t) = &mut self.phase {
            t.step += 1;
            let fraction = t.step as f32 / t.steps as f32;
            for (value, initial) in target.iter_mut().zip(&t.initial) {
                *value = initial + fraction * (*value - initial);
            }
            if t.step == t.steps {
                if matches!(self.phase, Phase::ToMotion(_)) {
                    // The first reference advance occurs on the next control tick.
                    self.phase = Phase::Motion(Instant::now() + Duration::from_secs_f64(period));
                    eprintln!("Motion tracking policy is active.");
                } else {
                    self.phase = Phase::Standing;
                    eprintln!("Standing policy resumed. Type MOTION to start the clip again.");
                }
            }
        }
        Ok(Output {
            log: json!({"phase": phase_name, "reference_time": reference_time,
            "policy": policy_log, "motor_target": target}),
            target,
            period: Duration::from_secs_f64(period),
        })
    }
}
