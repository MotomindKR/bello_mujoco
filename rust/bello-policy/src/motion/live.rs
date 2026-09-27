//! Task phases only: lease, command acknowledgements, and shutdown stay in control.

use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};
use glam::{DQuat, DVec3, Vec2};
use serde::Deserialize;
use serde_json::{Value, json};

use super::tilt::Envelope;
use super::{Feedback, Motion, ObservationMode, pose};
use crate::hardware::Hardware;
use crate::locomotion::{Feedback as HardwareFeedback, Locomotion};
use crate::target_limits::TargetLimits;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub standing_policy: PathBuf,
    pub pose: Option<pose::Config>,
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
    initial_gains: Vec<[f32; 2]>,
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
    pub motor_gains: Vec<[f32; 2]>,
    pub log: Value,
    pub period: Duration,
}

enum RootFeedback {
    Estimated {
        source: pose::Source,
        max_imu_orientation_error: f64,
    },
    Imu,
}

pub struct Live {
    motion: Motion,
    root_feedback: RootFeedback,
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
            a.robot == b.robot && a.coordinates == b.coordinates,
            "standing/motion robot or motor coordinates differ"
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
        let root_from_imu = motion.root_from_fixed_frame(standing.imu_frame())?;
        // Warm up the graph before acquiring a lease, not on the first motion tick.
        motion.policy.infer(&vec![0.; observation_size])?;
        let root_feedback = match (motion.settings.environment.observation_mode, &config.pose) {
            (ObservationMode::FullState, Some(pose)) => {
                motion.require_root_pose_frame(&pose.tracked_frame)?;
                RootFeedback::Estimated {
                    source: pose::Source::connect(pose.clone()).await?,
                    max_imu_orientation_error: pose.max_imu_orientation_error,
                }
            }
            (ObservationMode::Proprioceptive, None) => {
                eprintln!(
                    "Using IMU/joints without a pose service; absolute-height guards unavailable."
                );
                RootFeedback::Imu
            }
            (ObservationMode::FullState, None) => bail!("full-state motion requires motion.pose"),
            (ObservationMode::Proprioceptive, Some(_)) => {
                bail!("proprioceptive motion uses hardware IMU feedback; remove motion.pose")
            }
        };
        eprintln!(
            "Motion observation mode: {:?}",
            motion.settings.environment.observation_mode
        );
        let standing_limits = standing.policy.target_limits.clone();
        let transition_limits =
            standing_limits.transition_envelope(&motion.policy.target_limits)?;
        Ok(Self {
            motion,
            root_feedback,
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

    pub fn configure_hardware(&mut self, hardware: &Hardware) -> Result<()> {
        hardware.validate_policy_gains(&self.motion.policy.metadata)?;
        ensure!(
            hardware
                .groups
                .iter()
                .all(|group| group.control_rate_hz() + 1e-6 >= 1. / self.motion_period()),
            "hardware control rate is slower than the motion policy"
        );
        let standing = hardware.target_limits.clone();
        let motion = hardware.restrict_target_limits(self.motion_limits())?;
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
        self.feedback(feedback).map(|_| ())
    }

    fn feedback(&self, feedback: &HardwareFeedback) -> Result<Feedback> {
        let imu =
            super::rotation::checked_quaternion(DQuat::from_array(feedback.orientation_xyzw))?;
        let (position, orientation, linear_velocity, angular_velocity) = match &self.root_feedback {
            RootFeedback::Estimated {
                source,
                max_imu_orientation_error,
            } => {
                let pose = source.sample()?;
                ensure!(
                    (pose.orientation * self.root_from_imu)
                        .angle_between(imu)
                        .abs()
                        <= *max_imu_orientation_error,
                    "pose/IMU orientation mismatch"
                );
                (
                    Some(pose.position.to_array()),
                    pose.orientation,
                    Some(pose.linear_velocity.to_array()),
                    pose.angular_velocity,
                )
            }
            RootFeedback::Imu => (
                None,
                imu * self.root_from_imu.conjugate(),
                None,
                self.root_from_imu * DVec3::from_array(feedback.angular_velocity),
            ),
        };
        ensure!(
            angular_velocity.is_finite(),
            "nonfinite motion angular velocity"
        );
        Ok(Feedback {
            motor_position: feedback.motor_position.clone(),
            motor_velocity: feedback.motor_velocity.clone(),
            position,
            orientation_xyzw: orientation.to_array(),
            linear_velocity_world: linear_velocity,
            angular_velocity_local: angular_velocity.to_array(),
            reference_time: 0.,
            initialize: false,
        })
    }

    fn motor_gains(&self, standing: &Locomotion) -> Vec<[f32; 2]> {
        let (destination, transition) = match &self.phase {
            Phase::Standing => (&standing.policy.metadata, None),
            Phase::Motion(_) => (&self.motion.policy.metadata, None),
            Phase::ToStanding(t) => (&standing.policy.metadata, Some(t)),
            Phase::ToMotion(t) => (&self.motion.policy.metadata, Some(t)),
        };
        destination
            .kp
            .iter()
            .zip(&destination.kd)
            .enumerate()
            .map(|(i, (&kp, &kd))| {
                let destination = Vec2::new(kp, kd);
                transition
                    .map_or(destination, |t| {
                        Vec2::from_array(t.initial_gains[i])
                            .lerp(destination, t.step as f32 / t.steps as f32)
                    })
                    .to_array()
            })
            .collect()
    }

    fn transition(
        &self,
        initial: &[f64],
        standing: &Locomotion,
        period: f64,
        initial_tilt: Envelope,
    ) -> Transition {
        Transition {
            initial: initial.to_vec(),
            initial_gains: self.motor_gains(standing),
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
        let mut sample = self.feedback(feedback)?;
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
            self.phase = Phase::ToMotion(self.transition(
                previous_target,
                standing,
                self.motion_period(),
                tilt,
            ));
            self.motion.reset();
            self.interrupt.0.store(1, Ordering::SeqCst);
            eprintln!("Transitioning to motion tracking.");
        }
        let complete = matches!(self.phase, Phase::Motion(started)
            if self.config.start_time_seconds + now.saturating_duration_since(started).as_secs_f64() >= self.motion.duration());
        if complete || self.interrupt.0.load(Ordering::SeqCst) == 2 {
            self.phase = Phase::ToStanding(self.transition(
                previous_target,
                standing,
                standing_period,
                tilt,
            ));
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
                sample.reference_time = time;
                sample.initialize = initialize;
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
        let motor_gains = self.motor_gains(standing);
        Ok(Output {
            log: json!({"phase": phase_name, "reference_time": reference_time,
            "observation_mode": self.motion.settings.environment.observation_mode,
            "policy": policy_log, "motor_target": target, "motor_gains": motor_gains,
            "destination_target_limited": destination_target_limited,
            "tilt_error_radians": tilt_error, "tilt_error_limit_radians": tilt.max_error,
            "reference_gravity_imu": tilt.gravity.to_array()}),
            target,
            motor_gains,
            period: Duration::from_secs_f64(period),
        })
    }
}
