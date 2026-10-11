//! A bounded delayed reference window. No extrapolation or motor ownership.
use crate::{
    reference::{Sample, Source},
    sonic::{ReferenceFrame, Sonic},
};
use anyhow::{Result, ensure};
use glam::{DQuat, EulerRot};
use serde::Deserialize;
use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub policy: PathBuf,
    pub source: PathBuf,
    pub endpoint: String,
    pub rpc_timeout_seconds: f64,
    pub poll_period_seconds: f64,
    pub max_reference_age_ms: f64,
    pub max_sample_gap_seconds: f64,
    pub delay_seconds: f64,
    pub transition_seconds: f64,
    pub neutral_return_seconds: f64,
    pub neutral_joint_tolerance_radians: f64,
    pub neutral_settle_timeout_seconds: f64,
    pub standing_transition_seconds: f64,
    pub max_tilt_error_radians: f64,
}

#[derive(Clone, Default, serde::Serialize)]
pub struct Status {
    pub ready: bool,
    pub age_ms: f64,
    pub buffered_ms: f64,
    pub delay_ms: f64,
    pub message: String,
}

struct TimedFrame {
    at: Instant,
    frame: ReferenceFrame,
}

pub struct Buffer {
    frames: VecDeque<TimedFrame>,
    names: Vec<String>,
    last_sequence: Option<u64>,
    input_issue: Option<String>,
    gap: Duration,
    delay: Duration,
    max_age_ms: f64,
    dt: f64,
    pub status: Status,
}

impl Buffer {
    fn new(names: Vec<String>, config: &Config, future_dt: f64, period: f64) -> Result<Self> {
        for value in [
            config.rpc_timeout_seconds,
            config.poll_period_seconds,
            config.max_reference_age_ms,
            config.max_sample_gap_seconds,
            config.delay_seconds,
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 60_000.,
                "invalid tracking timing"
            );
        }
        ensure!(
            config.delay_seconds <= 10. && config.max_sample_gap_seconds <= 1.,
            "tracking buffer exceeds its bounded time window"
        );
        let horizon = 9. * future_dt + period;
        ensure!(
            config.delay_seconds >= horizon + config.max_sample_gap_seconds,
            "tracking delay must cover the SONIC future horizon and one sample gap"
        );
        ensure!(
            config.poll_period_seconds <= config.max_sample_gap_seconds,
            "tracking poll period exceeds sample gap"
        );
        Ok(Self {
            frames: VecDeque::new(),
            names,
            last_sequence: None,
            input_issue: None,
            gap: Duration::from_secs_f64(config.max_sample_gap_seconds),
            delay: Duration::from_secs_f64(config.delay_seconds),
            max_age_ms: config.max_reference_age_ms,
            dt: future_dt,
            status: Status {
                delay_ms: config.delay_seconds * 1000.,
                message: "Waiting for reference".into(),
                ..Status::default()
            },
        })
    }

    fn unavailable(&mut self, message: impl Into<String>) {
        self.frames.clear();
        self.last_sequence = None;
        self.input_issue = None;
        self.status.ready = false;
        self.status.buffered_ms = 0.;
        self.status.message = message.into();
    }

    fn ingest(&mut self, sample: Option<Sample>, now: Instant) -> Result<()> {
        let Some(sample) = sample else {
            self.input_issue = Some("Reference disconnected".into());
            return Ok(());
        };
        let message = sample.message;
        ensure!(
            message.source_age_ms.is_finite() && message.source_age_ms >= 0.,
            "invalid tracking source age"
        );
        self.status.age_ms = message.source_age_ms
            + now
                .saturating_duration_since(sample.requested)
                .as_secs_f64()
                * 1000.;
        if !message.available {
            self.input_issue = Some("Reference unavailable from producer".into());
            return Ok(());
        }
        if self.status.age_ms > self.max_age_ms {
            self.input_issue = Some(format!(
                "Reference stale: {:.1} ms (source {:.1} ms, limit {:.1} ms)",
                self.status.age_ms, message.source_age_ms, self.max_age_ms
            ));
            return Ok(());
        }
        self.input_issue = None;
        if self.last_sequence == Some(message.sequence) {
            return Ok(());
        }
        if self
            .last_sequence
            .is_some_and(|sequence| message.sequence < sequence)
        {
            self.unavailable("Reference restarted");
        }
        let body = message
            .full_body
            .ok_or_else(|| anyhow::anyhow!("Producer has no full-body reference"))?;
        ensure!(
            body.joint_names.len() == self.names.len()
                && body.joint_names.iter().collect::<BTreeSet<_>>() == self.names.iter().collect()
                && body.position.len() == self.names.len()
                && body.velocity.len() == self.names.len()
                && body
                    .position
                    .iter()
                    .chain(&body.velocity)
                    .all(|v| v.is_finite())
                && body.root_orientation_wxyz.len() == 4,
            "invalid full-body reference layout"
        );
        let q = &body.root_orientation_wxyz;
        let orientation = DQuat::from_xyzw(q[1], q[2], q[3], q[0]);
        ensure!(
            orientation.is_finite() && (0.9..=1.1).contains(&orientation.length()),
            "invalid tracking root orientation"
        );
        let indices = self
            .names
            .iter()
            .map(|name| body.joint_names.iter().position(|n| n == name).unwrap())
            .collect::<Vec<_>>();
        let frame = ReferenceFrame {
            position: indices.iter().map(|&i| body.position[i]).collect(),
            velocity: indices.iter().map(|&i| body.velocity[i]).collect(),
            orientation: orientation.normalize(),
        };
        let at = sample
            .requested
            .checked_sub(Duration::from_secs_f64(message.source_age_ms / 1000.))
            .ok_or_else(|| anyhow::anyhow!("invalid tracking timestamp"))?;
        if let Some(previous) = self.frames.back() {
            if at <= previous.at {
                // Source age is measured during the RPC. Timing jitter can put
                // a newer sequence slightly behind the last accepted sample.
                // Discard that sample without destroying valid buffered history.
                self.last_sequence = Some(message.sequence);
                return Ok(());
            }
            if at.duration_since(previous.at) > self.gap {
                self.unavailable("Refilling reference buffer");
            }
        }
        self.last_sequence = Some(message.sequence);
        self.frames.push_back(TimedFrame { at, frame });
        let oldest = now.checked_sub(self.delay + self.gap).unwrap_or(now);
        while self.frames.len() > 2 && self.frames[1].at < oldest {
            self.frames.pop_front();
        }
        ensure!(
            self.frames.len() <= 4096,
            "tracking reference rate exceeds buffer capacity"
        );
        Ok(())
    }

    fn sample(&self, at: Instant) -> Option<ReferenceFrame> {
        let mut previous = self.frames.front()?;
        if at < previous.at {
            return None;
        }
        for next in self.frames.iter().skip(1) {
            if at <= next.at {
                let t = at.duration_since(previous.at).as_secs_f64()
                    / next.at.duration_since(previous.at).as_secs_f64();
                let blend =
                    |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a + t * (b - a)).collect();
                return Some(ReferenceFrame {
                    position: blend(&previous.frame.position, &next.frame.position),
                    velocity: blend(&previous.frame.velocity, &next.frame.velocity),
                    orientation: previous.frame.orientation.slerp(next.frame.orientation, t),
                });
            }
            previous = next;
        }
        None
    }

    fn window(
        &mut self,
        now: Instant,
        period: f64,
    ) -> Option<(ReferenceFrame, Vec<ReferenceFrame>)> {
        self.status.buffered_ms = self
            .frames
            .front()
            .zip(self.frames.back())
            .map_or(0., |(a, b)| b.at.duration_since(a.at).as_secs_f64() * 1000.);
        let start = now.checked_sub(self.delay)?;
        let window = (|| {
            let initial = self.sample(start)?;
            let frames = (0..10)
                .map(|i| self.sample(start + Duration::from_secs_f64(period + i as f64 * self.dt)))
                .collect::<Option<Vec<_>>>()?;
            Some((initial, frames))
        })();
        self.status.ready = window.is_some();
        if self.status.ready {
            self.status.message = "Ready".into();
        } else if let Some(issue) = &self.input_issue {
            self.status.message.clone_from(issue);
        } else if !self.frames.is_empty() {
            self.status.message = "Buffering reference".into();
        }
        window
    }
}

struct NeutralReturn {
    initial: ReferenceFrame,
    neutral: Vec<f64>,
    started: Instant,
    duration: f64,
}

impl NeutralReturn {
    fn sample(&self, seconds: f64) -> ReferenceFrame {
        let t = (seconds / self.duration).clamp(0., 1.);
        let fraction = t * t * (3. - 2. * t);
        let rate = 6. * t * (1. - t) / self.duration;
        let yaw = DQuat::from_rotation_z(self.initial.orientation.to_euler(EulerRot::ZYX).0);
        ReferenceFrame {
            position: self
                .initial
                .position
                .iter()
                .zip(&self.neutral)
                .map(|(a, b)| a + fraction * (b - a))
                .collect(),
            velocity: self
                .initial
                .position
                .iter()
                .zip(&self.neutral)
                .map(|(a, b)| rate * (b - a))
                .collect(),
            orientation: self.initial.orientation.slerp(yaw, fraction),
        }
    }

    fn window(&self, now: Instant, period: f64, dt: f64) -> (ReferenceFrame, Vec<ReferenceFrame>) {
        let elapsed = now.saturating_duration_since(self.started).as_secs_f64();
        (
            self.sample(elapsed),
            (0..10)
                .map(|i| self.sample(elapsed + period + i as f64 * dt))
                .collect(),
        )
    }
}

pub struct Tracking {
    pub sonic: Sonic,
    source: Source,
    buffer: Buffer,
    window: Option<(ReferenceFrame, Vec<ReferenceFrame>)>,
    robot: String,
    config: Config,
    neutral: Vec<f64>,
    last_reference: Option<ReferenceFrame>,
    returning: Option<NeutralReturn>,
}
impl Tracking {
    pub fn new(sonic: Sonic, config: &Config) -> Result<Self> {
        for value in [
            config.neutral_return_seconds,
            config.neutral_joint_tolerance_radians,
            config.neutral_settle_timeout_seconds,
            config.standing_transition_seconds,
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 60.,
                "invalid SONIC return timing/tolerance"
            );
        }
        let robot = sonic.policy.metadata.robot.clone();
        let buffer = Buffer::new(
            sonic.joint_names().to_vec(),
            config,
            sonic.future_dt(),
            sonic.policy.metadata.control_timestep,
        )?;
        let source = Source::new(
            &config.endpoint,
            &robot,
            config.rpc_timeout_seconds,
            config.poll_period_seconds,
        )?;
        Ok(Self {
            sonic,
            source,
            buffer,
            window: None,
            robot,
            config: config.clone(),
            neutral: Vec::new(),
            last_reference: None,
            returning: None,
        })
    }
    pub fn configure_neutral(&mut self, motor_position: &[f64]) -> Result<()> {
        self.sonic.policy.target_limits.validate(motor_position)?;
        self.neutral = self.sonic.joint_positions(motor_position);
        Ok(())
    }
    pub fn reset(&mut self) {
        self.sonic.reset();
        self.returning = None;
        self.last_reference = None;
    }
    pub fn begin_return(&mut self) -> bool {
        let Some(initial) = self.last_reference.clone() else {
            return false;
        };
        self.returning = Some(NeutralReturn {
            initial,
            neutral: self.neutral.clone(),
            started: Instant::now(),
            duration: self.config.neutral_return_seconds,
        });
        true
    }
    pub fn return_complete(&self, motor_position: &[f64]) -> Result<bool> {
        let Some(returning) = &self.returning else {
            return Ok(false);
        };
        let elapsed = returning.started.elapsed().as_secs_f64();
        let error = self
            .sonic
            .joint_positions(motor_position)
            .iter()
            .zip(&self.neutral)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if elapsed >= returning.duration && error <= self.config.neutral_joint_tolerance_radians {
            return Ok(true);
        }
        ensure!(
            elapsed < returning.duration + self.config.neutral_settle_timeout_seconds,
            "SONIC neutral settling timed out: joint error {error:.4} rad"
        );
        Ok(false)
    }
    pub fn standing_transition_seconds(&self) -> f64 {
        self.config.standing_transition_seconds
    }
    pub fn refresh(&mut self) {
        let now = Instant::now();
        let result = self
            .source
            .latest()
            .map_err(anyhow::Error::msg)
            .and_then(|sample| {
                ensure!(
                    sample
                        .as_ref()
                        .is_none_or(|s| s.message.robot == self.robot),
                    "tracking robot mismatch"
                );
                self.buffer.ingest(sample, now)
            });
        if let Err(error) = result {
            self.buffer.unavailable(error.to_string());
        }
        self.window = self
            .buffer
            .window(now, self.sonic.policy.metadata.control_timestep);
    }
    pub fn status(&self) -> &Status {
        &self.buffer.status
    }
    pub fn reference_orientation(&self) -> DQuat {
        if let Some(returning) = &self.returning {
            return returning
                .sample(returning.started.elapsed().as_secs_f64())
                .orientation;
        }
        self.window
            .as_ref()
            .map_or(DQuat::IDENTITY, |(initial, _)| initial.orientation)
    }
    pub fn step(
        &mut self,
        feedback: &crate::locomotion::Feedback,
        initialize: bool,
    ) -> Result<crate::sonic::Step> {
        let generated = self.returning.as_ref().map(|returning| {
            returning.window(
                Instant::now(),
                self.sonic.policy.metadata.control_timestep,
                self.sonic.future_dt(),
            )
        });
        let (initial, frames) = generated
            .as_ref()
            .or(self.window.as_ref())
            .ok_or_else(|| anyhow::anyhow!(self.buffer.status.message.clone()))?;
        let step = self.sonic.step_reference(
            &crate::sonic::Feedback {
                motor_position: feedback.motor_position.clone(),
                motor_velocity: feedback.motor_velocity.clone(),
                orientation_xyzw: feedback.orientation_xyzw,
                angular_velocity_local: feedback.angular_velocity,
                reference_time: 0.,
                initialize,
            },
            initial.orientation,
            frames,
        )?;
        self.last_reference = Some(initial.clone());
        Ok(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::proto::{FullBodyReference, GetReferenceResponse};

    #[test]
    fn neutral_reference_is_bounded_upright_and_supplies_future_frames() {
        let started = Instant::now();
        let returning = NeutralReturn {
            initial: ReferenceFrame {
                position: vec![0.8, -0.4],
                velocity: vec![0.3, -0.2],
                orientation: DQuat::from_euler(EulerRot::ZYX, 0.7, 0.3, -0.2),
            },
            neutral: vec![0.2, 0.1],
            started,
            duration: 1.,
        };
        assert_eq!(returning.sample(0.).position, returning.initial.position);
        assert_eq!(returning.sample(0.).velocity, vec![0.; 2]);
        for i in 0..=100 {
            let seconds = i as f64 / 100.;
            let sample = returning.sample(seconds);
            assert!((0.2 - 1e-10..=0.8).contains(&sample.position[0]));
            assert!((-0.4..=0.1 + 1e-10).contains(&sample.position[1]));
            if i > 0 && i < 100 {
                let before = returning.sample(seconds - 1e-5);
                let after = returning.sample(seconds + 1e-5);
                for j in 0..2 {
                    assert!(
                        ((after.position[j] - before.position[j]) / 2e-5 - sample.velocity[j])
                            .abs()
                            < 1e-8
                    );
                }
            }
        }
        let (initial, future) = returning.window(started + Duration::from_millis(800), 0.02, 0.1);
        assert!(initial.position[0] > 0.2);
        assert_eq!(future.len(), 10);
        let last = future.last().unwrap();
        assert!((last.position[0] - 0.2).abs() < 1e-10);
        assert_eq!(last.velocity, vec![0.; 2]);
        assert!(
            last.orientation
                .angle_between(DQuat::from_rotation_z(0.7))
                .abs()
                < 1e-7
        );
        assert_eq!(returning.sample(3.).position, last.position);
    }

    fn config() -> Config {
        Config {
            policy: "policy.zip".into(),
            source: ".".into(),
            endpoint: "http://localhost:50056".into(),
            rpc_timeout_seconds: 0.1,
            poll_period_seconds: 0.02,
            max_reference_age_ms: 100.,
            max_sample_gap_seconds: 0.1,
            delay_seconds: 1.05,
            transition_seconds: 1.,
            neutral_return_seconds: 1.,
            neutral_joint_tolerance_radians: 0.15,
            neutral_settle_timeout_seconds: 2.,
            standing_transition_seconds: 0.2,
            max_tilt_error_radians: 0.5,
        }
    }
    fn sample(at: Instant, sequence: u64, value: f64) -> Sample {
        Sample {
            requested: at,
            message: GetReferenceResponse {
                robot: "synthetic".into(),
                available: true,
                sequence,
                full_body: Some(FullBodyReference {
                    joint_names: vec!["b".into(), "a".into()],
                    position: vec![-value, value],
                    velocity: vec![-2., 2.],
                    root_orientation_wxyz: vec![1., 0., 0., 0.],
                    ..Default::default()
                }),
                ..Default::default()
            },
        }
    }
    fn filled() -> (Buffer, Instant) {
        let mut buffer = Buffer::new(vec!["a".into(), "b".into()], &config(), 0.1, 0.02).unwrap();
        let start = Instant::now();
        for i in 0..=60 {
            let now = start + Duration::from_millis(i * 20);
            buffer
                .ingest(Some(sample(now, i, i as f64 * 0.02)), now)
                .unwrap();
        }
        (buffer, start + Duration::from_millis(1200))
    }
    #[test]
    fn delayed_window_interpolates_named_joints_without_extrapolation() {
        let (mut buffer, now) = filled();
        let (initial, frames) = buffer.window(now, 0.02).unwrap();
        assert!((initial.position[0] - 0.15).abs() < 1e-9);
        for (i, frame) in frames.iter().enumerate() {
            let expected = 0.17 + i as f64 * 0.1;
            assert!((frame.position[0] - expected).abs() < 1e-9);
            assert!((frame.position[1] + expected).abs() < 1e-9);
            assert_eq!(frame.velocity, [2., -2.]);
        }
        assert!(buffer.status.ready);
        assert!(buffer.window(now + Duration::from_secs(1), 0.02).is_none());
        assert!(!buffer.status.ready);
    }
    #[test]
    fn sustained_loss_staleness_restart_and_gaps_require_a_new_window() {
        for failure in ["loss", "stale", "restart", "gap"] {
            let (mut buffer, now) = filled();
            let later = now + Duration::from_millis(160);
            let mut next = sample(later, 61, 1.32);
            match failure {
                "stale" => next.message.source_age_ms = 101.,
                "restart" => next.message.sequence = 0,
                _ => {}
            }
            buffer
                .ingest((failure != "loss").then_some(next), later)
                .unwrap();
            assert!(buffer.window(later, 0.02).is_none(), "{failure}");
        }
    }
    #[test]
    fn brief_input_loss_uses_only_the_already_validated_window() {
        for failure in ["loss", "stale", "unavailable"] {
            let (mut buffer, now) = filled();
            let count = buffer.frames.len();
            for ms in [20, 40, 60] {
                let later = now + Duration::from_millis(ms);
                let mut next = sample(later, 61, 99.);
                next.message.source_age_ms = if failure == "stale" { 101. } else { 0. };
                next.message.available = failure != "unavailable";
                buffer
                    .ingest((failure != "loss").then_some(next), later)
                    .unwrap();
                assert_eq!(buffer.frames.len(), count);
                assert_eq!(buffer.frames.back().unwrap().frame.position[0], 1.2);
                assert!(buffer.window(later, 0.02).is_some(), "{failure}");
            }
            let later = now + Duration::from_millis(80);
            buffer.ingest(Some(sample(later, 62, 1.28)), later).unwrap();
            assert!(buffer.window(later, 0.02).is_some());
            assert!(
                buffer
                    .window(later + Duration::from_millis(140), 0.02)
                    .is_none()
            );
        }
    }
    #[test]
    fn duplicate_samples_do_not_extend_the_window() {
        let (mut buffer, now) = filled();
        let count = buffer.frames.len();
        buffer
            .ingest(Some(sample(now + Duration::from_millis(20), 60, 8.)), now)
            .unwrap();
        assert_eq!(buffer.frames.len(), count);
        assert_eq!(buffer.frames.back().unwrap().frame.position[0], 1.2);
    }
    #[test]
    fn age_jitter_discards_out_of_order_samples_without_resetting_history() {
        for age_ms in [20., 25.] {
            let (mut buffer, now) = filled();
            let count = buffer.frames.len();
            let later = now + Duration::from_millis(20);
            let mut jittered = sample(later, 61, 99.);
            jittered.message.source_age_ms = age_ms;
            buffer.ingest(Some(jittered), later).unwrap();
            assert_eq!(buffer.frames.len(), count);
            assert_eq!(buffer.frames.back().unwrap().frame.position[0], 1.2);
            assert!(buffer.window(later, 0.02).is_some());

            // Re-reading the rejected sequence must not append it at a new time.
            let later = now + Duration::from_millis(40);
            buffer.ingest(Some(sample(later, 61, 99.)), later).unwrap();
            assert_eq!(buffer.frames.len(), count);
            buffer.ingest(Some(sample(later, 62, 1.24)), later).unwrap();
            assert!(buffer.window(later, 0.02).is_some());
            assert_eq!(buffer.frames.back().unwrap().frame.position[0], 1.24);

            // History cannot provide future frames indefinitely when input stops.
            assert!(
                buffer
                    .window(later + Duration::from_secs(1), 0.02)
                    .is_none()
            );
        }
    }
    #[test]
    fn malformed_layout_and_short_delay_are_rejected() {
        let mut cfg = config();
        cfg.delay_seconds = 0.9;
        assert!(Buffer::new(vec!["a".into()], &cfg, 0.1, 0.02).is_err());
        let (mut buffer, now) = filled();
        let mut next = sample(now, 61, 1.);
        next.message.full_body.as_mut().unwrap().joint_names = vec!["a".into(), "a".into()];
        assert!(buffer.ingest(Some(next), now).is_err());
    }
    #[test]
    fn retention_is_bounded_during_long_sessions() {
        let (mut buffer, now) = filled();
        for i in 1..10000 {
            let next = now + Duration::from_millis(i * 20);
            buffer.ingest(Some(sample(next, 60 + i, 0.)), next).unwrap();
        }
        assert!(buffer.frames.len() < 65);
    }
}
