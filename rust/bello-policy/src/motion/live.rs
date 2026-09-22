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

use super::tilt::Envelope;
use super::{Feedback, Motion, pose};
use crate::locomotion::{Feedback as HardwareFeedback, Locomotion};
use crate::target_limits::TargetLimits;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub standing_policy: PathBuf,
    pub pose: pose::Config,
    pub start_time_seconds: f64,
    pub transition_seconds: f64,
    pub max_tilt_error_radians: f64,
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
    initial: Vec<f64>,
    step: usize,
    steps: usize,
    initial_tilt: Envelope,
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
    standing_limits: TargetLimits,
    transition_limits: TargetLimits,
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
            config.start_time_seconds.is_finite()
                && config.start_time_seconds >= 0.
                && config.start_time_seconds < motion.duration(),
            "invalid motion start time"
        );
        for value in [
            config.transition_seconds,
            f64::from(config.max_raw_action_abs),
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 86400.,
                "invalid motion control limit/timing"
            );
        }
        ensure!(
            config.max_tilt_error_radians.is_finite()
                && config.max_tilt_error_radians > 0.
                && config.max_tilt_error_radians < std::f64::consts::PI,
            "motion max_tilt_error_radians must be in (0, pi)"
        );
        let observation_size = a.observation_size;
        motion.require_root_pose_frame(&config.pose.tracked_frame)?;
        let root_from_imu = motion.root_from_fixed_frame(standing.imu_frame())?;
        // Warm up the graph before acquiring a lease, not on the first motion tick.
        motion.policy.infer(&vec![0.; observation_size])?;
        let poses = pose::Source::connect(config.pose.clone()).await?;
        let standing_limits = standing.policy.target_limits.clone();
        let transition_limits =
            standing_limits.transition_envelope(&motion.policy.target_limits)?;
        Ok(Self {
            motion,
            poses,
            config,
            root_from_imu,
            phase: Phase::Standing,
            standing_limits,
            transition_limits,
            interrupt: Interrupt::default(),
        })
    }

    pub fn motion_period(&self) -> f64 {
        self.motion.policy.metadata.control_timestep
    }

    pub fn motion_limits(&self) -> &TargetLimits {
        &self.motion.policy.target_limits
    }

    pub fn configure_target_limits(
        &mut self,
        standing: TargetLimits,
        motion: TargetLimits,
    ) -> Result<()> {
        self.transition_limits = standing.transition_envelope(&motion)?;
        self.standing_limits = standing;
        self.motion.policy.target_limits = motion;
        Ok(())
    }

    pub fn target_limits(&self) -> &TargetLimits {
        match self.phase {
            Phase::Standing => &self.standing_limits,
            Phase::Motion(_) => self.motion_limits(),
            Phase::ToMotion(_) | Phase::ToStanding(_) => &self.transition_limits,
        }
    }

    pub fn absolute_tilt_limit(&self, standing: f64) -> Option<f64> {
        if matches!(self.phase, Phase::Standing) {
            Some(standing)
        } else {
            None
        }
    }

    fn reference_tilt(&self, time: f64) -> Envelope {
        let sample = self.motion.clip.sample(time);
        let root = super::rotation::from_wxyz(sample.qpos[3..7].try_into().unwrap());
        Envelope::new(
            root * self.root_from_imu,
            self.config.max_tilt_error_radians,
        )
    }

    fn tilt_envelope(&self, now: Instant, standing_limit: f64) -> Envelope {
        let standing = Envelope::new(DQuat::IDENTITY, standing_limit);
        match &self.phase {
            Phase::Standing => standing,
            Phase::Motion(started) => self.reference_tilt(
                self.config.start_time_seconds
                    + now.saturating_duration_since(*started).as_secs_f64(),
            ),
            Phase::ToMotion(t) => t.initial_tilt.blend(
                self.reference_tilt(self.config.start_time_seconds),
                t.step as f64 / t.steps as f64,
            ),
            Phase::ToStanding(t) => t
                .initial_tilt
                .blend(standing, t.step as f64 / t.steps as f64),
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

    fn transition(&self, initial: &[f64], period: f64, initial_tilt: Envelope) -> Transition {
        Transition {
            initial: initial.to_vec(),
            step: 0,
            steps: (self.config.transition_seconds / period).ceil() as usize,
            initial_tilt,
        }
    }

    pub fn step(
        &mut self,
        standing: &mut Locomotion,
        feedback: &HardwareFeedback,
        request_motion: bool,
        previous_target: &[f64],
        standing_tilt_limit: f64,
    ) -> Result<Output> {
        let pose = self.pose(feedback)?;
        let now = Instant::now();
        // Validate before phase changes: interrupts must not reset an unsafe pose
        // into a newly permissive envelope. The endpoint is checked before handover.
        let tilt = self.tilt_envelope(now, standing_tilt_limit);
        let tilt_error = tilt.check(DQuat::from_array(feedback.orientation_xyzw))?;
        let standing_period = standing.policy.metadata.control_timestep;
        // A scheduled blend can finish before step limiting reaches the destination.
        // Switch envelopes only after the last acknowledged command is feasible there.
        match &self.phase {
            Phase::ToMotion(t)
                if t.step >= t.steps && self.motion_limits().validate(previous_target).is_ok() =>
            {
                self.phase = Phase::Motion(now);
                eprintln!("Motion tracking policy is active.");
            }
            Phase::ToStanding(t)
                if t.step >= t.steps && self.standing_limits.validate(previous_target).is_ok() =>
            {
                self.phase = Phase::Standing;
                eprintln!("Standing policy resumed. Type MOTION to start the clip again.");
            }
            _ => {}
        }
        if request_motion {
            ensure!(
                matches!(self.phase, Phase::Standing),
                "MOTION is only accepted while standing"
            );
            self.phase =
                Phase::ToMotion(self.transition(previous_target, self.motion_period(), tilt));
            self.motion.reset();
            self.interrupt.0.store(1, Ordering::SeqCst);
            eprintln!("Transitioning to motion tracking.");
        }
        let complete = matches!(self.phase, Phase::Motion(started)
            if self.config.start_time_seconds + now.saturating_duration_since(started).as_secs_f64() >= self.motion.duration());
        if complete || self.interrupt.0.load(Ordering::SeqCst) == 2 {
            self.phase = Phase::ToStanding(self.transition(previous_target, standing_period, tilt));
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
        let destination = match self.phase {
            Phase::Standing | Phase::ToStanding(_) => &self.standing_limits,
            Phase::Motion(_) | Phase::ToMotion(_) => self.motion_limits(),
        };
        let mut target: Vec<f64> = target.iter().map(|v| *v as f64).collect();
        let mut destination_target_limited = false;
        if matches!(self.phase, Phase::ToMotion(_) | Phase::ToStanding(_)) {
            let projected = destination.project(&target)?;
            destination_target_limited = projected != target;
            target = projected;
        }
        if let Phase::ToMotion(t) | Phase::ToStanding(t) = &mut self.phase {
            t.step = (t.step + 1).min(t.steps);
            let fraction = t.step as f64 / t.steps as f64;
            for (value, initial) in target.iter_mut().zip(&t.initial) {
                *value = initial + fraction * (*value - initial);
            }
        }
        let target: Vec<f32> = target.into_iter().map(|v| v as f32).collect();
        Ok(Output {
            log: json!({"phase": phase_name, "reference_time": reference_time,
            "policy": policy_log, "motor_target": target,
            "destination_target_limited": destination_target_limited,
            "tilt_error_radians": tilt_error, "tilt_error_limit_radians": tilt.max_error,
            "reference_gravity_imu": tilt.gravity.to_array()}),
            target,
            period: Duration::from_secs_f64(period),
        })
    }
}
