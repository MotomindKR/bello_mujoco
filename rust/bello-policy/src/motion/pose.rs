//! Bounded latest-pose stream. Loss or malformed feedback stops motion control.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use bello_pose_api::{WatchPoseRequest, WatchPoseResponse, pose_service_client::PoseServiceClient};
use glam::{DQuat, DVec3};
use serde::Deserialize;
use tokio::sync::watch;

use super::rotation::checked_quaternion;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub endpoint: String,
    pub reference_frame: String,
    pub tracked_frame: String,
    pub timeout_seconds: f64,
    pub max_age_ms: f64,
    pub max_linear_velocity: f64,
    pub max_angular_velocity: f64,
    pub max_imu_orientation_error: f64,
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.reference_frame.is_empty() && !self.tracked_frame.is_empty(),
            "missing pose frame name"
        );
        for value in [
            self.timeout_seconds,
            self.max_age_ms,
            self.max_linear_velocity,
            self.max_angular_velocity,
            self.max_imu_orientation_error,
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 86400.,
                "invalid pose timeout/safety limit"
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Pose {
    pub position: DVec3,
    pub orientation: DQuat,
    pub linear_velocity: DVec3,
    pub angular_velocity: DVec3,
    received: Instant,
    age_ms: f64,
}

#[derive(Default)]
struct Decoder {
    previous: Option<(u64, u64)>,
    clock_anchor: Option<(u64, Instant, f64)>,
}

impl Decoder {
    fn decode(
        &mut self,
        message: WatchPoseResponse,
        config: &Config,
        now: Instant,
    ) -> Result<Pose> {
        ensure!(
            message.reference_frame == config.reference_frame
                && message.tracked_frame == config.tracked_frame,
            "pose frame mismatch"
        );
        ensure!(
            message.sample_monotonic_ns > 0,
            "missing pose source timestamp"
        );
        if let Some((sequence, stamp)) = self.previous {
            ensure!(
                message.sequence > sequence && message.sample_monotonic_ns > stamp,
                "pose sequence/timestamp did not increase"
            );
        }
        let position = message
            .translation_metres
            .context("missing pose position")?;
        let position = DVec3::new(position.x, position.y, position.z);
        ensure!(position.is_finite(), "nonfinite pose position");
        let orientation = message
            .orientation_xyzw
            .context("missing pose orientation")?;
        let orientation = checked_quaternion(DQuat::from_xyzw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ))?;
        let linear = message
            .linear_velocity_metres_per_second
            .context("missing pose linear velocity")?;
        let angular = message
            .angular_velocity_radians_per_second
            .context("missing pose angular velocity")?;
        let linear_velocity = DVec3::new(linear.x, linear.y, linear.z);
        let angular_velocity = DVec3::new(angular.x, angular.y, angular.z);
        ensure!(
            linear_velocity.is_finite() && linear_velocity.length() <= config.max_linear_velocity,
            "pose linear velocity exceeds safety limit"
        );
        ensure!(
            angular_velocity.is_finite()
                && angular_velocity.length() <= config.max_angular_velocity,
            "pose angular velocity exceeds safety limit"
        );
        // Compare elapsed source/receiver time, never absolute clocks on different hosts.
        // Retain the least-delayed observation so accumulating stream backlog ages out.
        let mut age_ms = message.sample_age_ms as f64;
        if let Some((stamp, received, initial_age_ms)) = self.clock_anchor {
            let elapsed_source = Duration::from_nanos(message.sample_monotonic_ns - stamp);
            let lag = now.duration_since(received).as_secs_f64() - elapsed_source.as_secs_f64();
            let projected_age_ms = initial_age_ms + lag * 1000.;
            if projected_age_ms < age_ms {
                self.clock_anchor = Some((message.sample_monotonic_ns, now, age_ms));
            }
            age_ms = age_ms.max(projected_age_ms);
        } else {
            self.clock_anchor = Some((message.sample_monotonic_ns, now, age_ms));
        }
        ensure!(age_ms <= config.max_age_ms, "stale pose feedback");
        self.previous = Some((message.sequence, message.sample_monotonic_ns));
        Ok(Pose {
            position,
            orientation,
            linear_velocity,
            angular_velocity,
            received: now,
            age_ms,
        })
    }
}

pub struct Source {
    latest: watch::Receiver<Option<Result<Pose, String>>>,
    task: tokio::task::JoinHandle<()>,
    config: Config,
}

impl Source {
    pub async fn connect(config: Config) -> Result<Self> {
        config.validate()?;
        let timeout = Duration::from_secs_f64(config.timeout_seconds);
        let connection = async {
            let channel = tonic::transport::Endpoint::from_shared(config.endpoint.clone())?
                .connect_timeout(timeout)
                .connect()
                .await?;
            let mut client = PoseServiceClient::new(channel).max_decoding_message_size(16 * 1024);
            Ok::<_, anyhow::Error>(
                client
                    .watch_pose(WatchPoseRequest {
                        reference_frame: config.reference_frame.clone(),
                        tracked_frame: config.tracked_frame.clone(),
                    })
                    .await?
                    .into_inner(),
            )
        };
        let mut stream = tokio::time::timeout(timeout, connection)
            .await
            .context("pose connection timed out")??;
        let (tx, latest) = watch::channel(None);
        let settings = config.clone();
        let task = tokio::spawn(async move {
            let mut decoder = Decoder::default();
            loop {
                let next = async {
                    let message = stream.message().await?.context("pose stream closed")?;
                    decoder.decode(message, &settings, Instant::now())
                }
                .await;
                let failed = next.is_err();
                tx.send_replace(Some(next.map_err(|error| error.to_string())));
                if failed {
                    break;
                }
            }
        });
        let mut source = Self {
            latest,
            task,
            config,
        };
        tokio::time::timeout(timeout, async {
            while source.latest.borrow().is_none() {
                source.latest.changed().await?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("waiting for first pose timed out")??;
        source.sample()?;
        Ok(source)
    }

    pub fn sample(&self) -> Result<Pose> {
        let sample = self
            .latest
            .borrow()
            .clone()
            .context("pose is not available")?
            .map_err(anyhow::Error::msg)?;
        ensure!(
            sample.age_ms + sample.received.elapsed().as_secs_f64() * 1000.
                <= self.config.max_age_ms,
            "stale pose feedback"
        );
        Ok(sample)
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            endpoint: "http://localhost:1".into(),
            reference_frame: "world".into(),
            tracked_frame: "base".into(),
            timeout_seconds: 1.,
            max_age_ms: 80.,
            max_linear_velocity: 5.,
            max_angular_velocity: 10.,
            max_imu_orientation_error: 0.2,
        }
    }

    fn message(sequence: u64, stamp: u64) -> WatchPoseResponse {
        WatchPoseResponse {
            reference_frame: "world".into(),
            tracked_frame: "base".into(),
            sequence,
            sample_monotonic_ns: stamp,
            translation_metres: Some(bello_pose_api::Translation::default()),
            orientation_xyzw: Some(bello_pose_api::Orientation {
                w: 1.,
                ..Default::default()
            }),
            linear_velocity_metres_per_second: Some(bello_pose_api::LinearVelocity::default()),
            angular_velocity_radians_per_second: Some(bello_pose_api::AngularVelocity::default()),
            ..Default::default()
        }
    }

    #[test]
    fn accumulated_network_backlog_adds_to_reported_source_age() {
        let mut decoder = Decoder::default();
        let now = Instant::now();
        let mut first = message(1, 1_000_000_000);
        first.sample_age_ms = 50;
        decoder.decode(first, &config(), now).unwrap();
        let next = message(2, 1_010_000_000);
        // 50 ms source age + 50 ms excess transit, even if the next report says zero.
        let error = decoder
            .decode(next, &config(), now + Duration::from_millis(60))
            .err()
            .unwrap();
        assert!(error.to_string().contains("stale"));
    }

    #[test]
    fn malformed_live_poses_are_not_replaced_with_defaults() {
        let now = Instant::now();
        let mut decoder = Decoder::default();
        decoder.decode(message(1, 1), &config(), now).unwrap();
        assert!(decoder.decode(message(1, 1), &config(), now).is_err());
        let mut missing = message(2, 2);
        missing.linear_velocity_metres_per_second = None;
        assert!(decoder.decode(missing, &config(), now).is_err());
        let mut fast = message(2, 2);
        fast.angular_velocity_radians_per_second.as_mut().unwrap().x = 11.;
        assert!(decoder.decode(fast, &config(), now).is_err());
    }
}
