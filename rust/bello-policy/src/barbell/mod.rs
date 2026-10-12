//! Standing control, external arm targets, hand ownership, and motion as one mode.
pub mod hands;
pub mod sequence;
use crate::{
    Policy,
    hardware::Hardware,
    locomotion::{Feedback, Locomotion},
    motion::{
        Motion,
        live::{Interrupt, Live, Output},
    },
    target_limits::TargetLimits,
};
use anyhow::{Result, ensure};
use sequence::{Phase, Sequence, Timing};
use serde::Deserialize;
use serde_json::json;
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub policy: PathBuf,
    pub source: PathBuf,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub standing: Bundle,
    pub motion: Bundle,
    #[serde(default)]
    pub allow_unscreened_reference: bool,
    pub motion_settings: crate::motion::live::Config,
    pub arm_poses: PathBuf,
    pub timing: Timing,
    pub grip_adjustment_per_second: f64,
    pub hands: hands::Config,
}
#[derive(Deserialize)]
struct Poses {
    coordinates: Vec<String>,
    normal: Vec<f64>,
    unloaded_forward: Vec<f64>,
    loaded_forward: Vec<f64>,
}
pub struct Runtime {
    standing: Locomotion,
    motion: Live,
    poses: Poses,
    timing: Timing,
    grip_adjustment_per_second: f64,
    hand_config: hands::Config,
    hands: Option<hands::Session>,
    sequence: Sequence,
    initialized: bool,
    limits: TargetLimits,
    pub interrupt: Interrupt,
}
impl Runtime {
    pub async fn load(config: Config, primary: &Locomotion) -> Result<Self> {
        config.timing.validate()?;
        config.hands.validate()?;
        ensure!(
            config.grip_adjustment_per_second.is_finite() && config.grip_adjustment_per_second > 0.,
            "barbell grip_adjustment_per_second must be positive and finite"
        );
        ensure!(
            config.motion_settings.start_time_seconds == 0.,
            "barbell motion must start at zero"
        );
        ensure!(
            (config.motion_settings.transition_seconds - config.timing.policy_blend_seconds).abs()
                < 1e-9,
            "barbell blend timings differ"
        );
        let mut standing = Locomotion::new(Policy::load(
            &config.standing.policy,
            &config.standing.source,
        )?)?;
        ensure!(
            standing.policy.metadata.coordinates == primary.policy.metadata.coordinates
                && standing.policy.metadata.robot == primary.policy.metadata.robot
                && standing.imu_frame() == primary.imu_frame(),
            "barbell standing robot/coordinates/IMU differ"
        );
        for key in ["joint_names", "joint_from_motor", "policy_from_hardware"] {
            ensure!(
                standing.policy.document.get(key) == primary.policy.document.get(key),
                "barbell standing {key} differs"
            );
        }
        let poses: Poses = serde_json::from_reader(std::fs::File::open(config.arm_poses)?)?;
        ensure!(
            poses.coordinates == standing.policy.metadata.coordinates,
            "barbell arm pose coordinate order differs"
        );
        let n = poses.coordinates.len();
        for pose in [
            &poses.normal,
            &poses.unloaded_forward,
            &poses.loaded_forward,
        ] {
            ensure!(pose.len() == n, "barbell arm pose dimensions differ");
            standing.set_arm_target(pose)?;
        }
        for &i in standing.held_indices() {
            ensure!(
                !standing.arm_conditioned()
                    || (poses.loaded_forward[i]
                        - f64::from(standing.policy.metadata.default_position[i]))
                    .abs()
                        < 1e-6,
                "barbell loaded arm target differs from trained preload"
            );
            ensure!(
                (poses.normal[i] - f64::from(primary.policy.metadata.default_position[i])).abs()
                    < 1e-6,
                "barbell normal arm pose differs from primary standing"
            );
        }
        standing.set_arm_target(&poses.normal)?;
        let size = standing.policy.metadata.observation_size;
        standing.policy.infer(&vec![0.; size])?;
        let motion = Live::connect(
            Motion::load(
                Policy::load(&config.motion.policy, &config.motion.source)?,
                config.allow_unscreened_reference,
            )?,
            &standing,
            config.motion_settings,
        )
        .await?;
        ensure!(
            (standing.policy.metadata.control_timestep - primary.policy.metadata.control_timestep)
                .abs()
                < 1e-9
                && (motion.motion_period() - standing.policy.metadata.control_timestep).abs()
                    < 1e-9,
            "barbell policies must share one control period"
        );
        ensure!(
            config.timing.motion_seconds <= motion.duration(),
            "barbell duration exceeds bundled motion"
        );
        let limits = primary
            .policy
            .target_limits
            .transition_envelope(&standing.policy.target_limits)?
            .transition_envelope(motion.motion_limits())?;
        Ok(Self {
            standing,
            motion,
            poses,
            sequence: Sequence::new(config.timing.clone()),
            timing: config.timing,
            grip_adjustment_per_second: config.grip_adjustment_per_second,
            hand_config: config.hands,
            hands: None,
            initialized: false,
            limits,
            interrupt: Interrupt::default(),
        })
    }
    pub fn configure_hardware(&mut self, hardware: &Hardware) -> Result<()> {
        hardware.validate_policy_gains(&self.standing.policy.metadata)?;
        self.standing.policy.target_limits =
            hardware.restrict_target_limits(&self.standing.policy.target_limits)?;
        for pose in [
            &self.poses.normal,
            &self.poses.unloaded_forward,
            &self.poses.loaded_forward,
        ] {
            self.standing.policy.target_limits.validate(pose)?;
        }
        self.motion.configure_hardware(hardware)?;
        self.limits = hardware
            .target_limits
            .transition_envelope(&self.standing.policy.target_limits)?
            .transition_envelope(self.motion.motion_limits())?;
        Ok(())
    }
    pub fn check_feedback(&self, sample: &Feedback) -> Result<()> {
        self.motion.check_feedback(sample)
    }
    pub fn start(&mut self) -> Result<()> {
        ensure!(self.hands.is_none(), "barbell session already active");
        self.sequence = Sequence::new(self.timing.clone());
        self.standing.reset();
        self.motion.reset();
        self.initialized = false;
        self.interrupt.arm();
        self.hands = Some(hands::Session::start(self.hand_config.clone()));
        Ok(())
    }
    pub fn stop(&mut self, reason: &str) {
        self.sequence.stop(reason);
    }
    pub fn confirm(&mut self, event: &str, phase_id: u64) -> Result<()> {
        let status = self.status()?;
        ensure!(status.confirmation_ready, "barbell hand feedback not ready");
        self.sequence.confirm(event, phase_id)
    }
    pub fn status(&self) -> Result<crate::operator::proto::BarbellStatus> {
        if self.hands.as_ref().is_some_and(|s| s.closing()) {
            return Ok(crate::operator::proto::BarbellStatus {
                message: "Releasing hand leases".into(),
                ..Default::default()
            });
        }
        let state = self.hands.as_ref().map(|s| s.state()).transpose()?;
        let cmd = &self.sequence.command;
        let ready = state
            .as_ref()
            .is_some_and(|s| s.ready && s.closure == cmd.closure);
        Ok(crate::operator::proto::BarbellStatus {
            phase_id: cmd.phase_id,
            awaiting: self.sequence.awaiting().into(),
            confirmation_ready: ready && (cmd.phase != Phase::Close || cmd.closure > 0.),
            hands_owned: state.as_ref().is_some_and(|s| s.ready),
            grip_closure: cmd.closure,
            grip_adjustable: cmd.phase == Phase::Close && state.as_ref().is_some_and(|s| s.ready),
            load_fraction: cmd.load,
            motion_weight: cmd.weight,
            message: if !cmd.reason.is_empty() {
                cmd.reason.clone()
            } else if cmd.phase == Phase::Close {
                "Set grip with left stick; center stick before confirming secure grasp".into()
            } else if !ready {
                "Waiting for hand ownership / fresh feedback".into()
            } else {
                "Both hand leases active; handoff confirmations are manual".into()
            },
        })
    }
    pub fn phase(&self) -> &str {
        if self
            .hands
            .as_ref()
            .and_then(|s| s.state().ok())
            .is_some_and(|s| !s.ready)
        {
            "acquiring_hands"
        } else {
            self.sequence.command.phase.name()
        }
    }
    pub fn progress(&self) -> f64 {
        self.sequence.command.progress
    }
    pub fn complete(&self) -> bool {
        self.sequence.command.phase == Phase::Complete
    }
    pub fn limits(&self) -> &TargetLimits {
        &self.limits
    }
    pub async fn release_complete(&mut self) -> Result<bool> {
        let session = self.hands.as_mut().expect("barbell started");
        session.begin_close();
        if !session.finished() {
            return Ok(false);
        }
        self.close().await?;
        Ok(true)
    }
    pub async fn close(&mut self) -> Result<()> {
        if let Some(mut s) = self.hands.take() {
            s.close().await?;
        }
        Ok(())
    }
    pub fn step(
        &mut self,
        primary: &mut Locomotion,
        feedback: &Feedback,
        tilt: f64,
        grip_axis: f32,
    ) -> Result<Output> {
        if self.interrupt.returning() {
            self.stop("Operator requested return");
        }
        let session = self.hands.as_ref().expect("barbell started");
        if session.closing() {
            crate::motion::tilt::Envelope::new(glam::DQuat::IDENTITY, tilt)
                .check(glam::DQuat::from_array(feedback.orientation_xyzw))?;
            let step = primary.step(feedback)?;
            return Ok(Output {
                target: step.motor_target.clone(),
                motor_gains: gains(primary),
                log: json!({"phase":"releasing_hands","policy":step}),
                period: Duration::from_secs_f64(primary.policy.metadata.control_timestep),
            });
        }
        let state = session.state()?;
        let dt = self.standing.policy.metadata.control_timestep;
        if !state.ready {
            crate::motion::tilt::Envelope::new(glam::DQuat::IDENTITY, tilt)
                .check(glam::DQuat::from_array(feedback.orientation_xyzw))?;
            let step = primary.step(feedback)?;
            return Ok(Output {
                target: step.motor_target.clone(),
                motor_gains: gains(primary),
                log: json!({"phase":"acquiring_hands","policy":step}),
                period: Duration::from_secs_f64(dt),
            });
        }
        let base = primary.step(feedback)?;
        self.sequence
            .adjust_grip(f64::from(grip_axis) * self.grip_adjustment_per_second * dt);
        let cmd = self.sequence.advance(dt);
        session.command(self.sequence.command.closure);
        let mut arm = self.poses.normal.clone();
        for &i in self.standing.held_indices() {
            let forward = self.poses.unloaded_forward[i] * (1. - cmd.load)
                + self.poses.loaded_forward[i] * cmd.load;
            arm[i] = self.poses.normal[i] * (1. - cmd.arm) + forward * cmd.arm;
        }
        self.standing.set_arm_target(&arm)?;
        let mut output = if matches!(
            cmd.phase,
            Phase::EnterMotion | Phase::Motion | Phase::Return
        ) {
            let output = self.motion.coordinated_step(
                &mut self.standing,
                feedback,
                cmd.motion_time,
                cmd.weight,
                !self.initialized,
                tilt,
            )?;
            self.initialized = true;
            output
        } else {
            crate::motion::tilt::Envelope::new(glam::DQuat::IDENTITY, tilt)
                .check(glam::DQuat::from_array(feedback.orientation_xyzw))?;
            let step = self.standing.step(feedback)?;
            Output {
                target: step.motor_target.clone(),
                motor_gains: gains(&self.standing),
                log: json!({"standing":step}),
                period: Duration::from_secs_f64(dt),
            }
        };
        if matches!(cmd.phase, Phase::Stand | Phase::Finish | Phase::Complete) {
            let weight = if cmd.phase == Phase::Stand {
                cmd.progress
            } else if cmd.phase == Phase::Complete {
                0.
            } else {
                1. - cmd.progress
            } as f32;
            for (target, base) in output.target.iter_mut().zip(&base.motor_target) {
                *target = *target * weight + base * (1. - weight);
            }
            for (target, base) in output.motor_gains.iter_mut().zip(gains(primary)) {
                for i in 0..2 {
                    target[i] = target[i] * weight + base[i] * (1. - weight);
                }
            }
        }
        output.log = json!({"sequence":cmd,"policy":output.log});
        Ok(output)
    }
}
fn gains(policy: &Locomotion) -> Vec<[f32; 2]> {
    policy
        .policy
        .metadata
        .kp
        .iter()
        .zip(&policy.policy.metadata.kd)
        .map(|(&kp, &kd)| [kp, kd])
        .collect()
}
