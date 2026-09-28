//! Read-only commissioning against the unchanged hardware gRPC API.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use bello_hardware_client::{
    ConnectOptions, DeviceState, GroupState, MotorGroup, Robot, raw::proto,
};
use glam::DVec3;
use serde::Deserialize;

use crate::PolicyMetadata;
use crate::locomotion::{Feedback, Locomotion, projected_gravity};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub control: Option<crate::control::Config>,
    pub motion: Option<crate::motion::live::Config>,
    pub endpoint: String,
    pub groups: Vec<String>,
    pub imu_name: String,
    pub connection_timeout_seconds: f64,
    pub telemetry_timeout_seconds: f64,
    pub max_motor_age_ms: f64,
    pub max_imu_age_ms: f64,
    pub max_group_age_ms: f64,
    pub max_state_sample_span_ms: f64,
    #[serde(default)]
    pub startup_feedback_margin_radians: f64,
    pub max_tilt_radians: f64,
    pub max_angular_velocity: f64,
    pub max_linear_acceleration: f64,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.groups.is_empty()
                && self.groups.iter().all(|s| !s.is_empty())
                && self.groups.iter().collect::<BTreeSet<_>>().len() == self.groups.len(),
            "invalid hardware groups"
        );
        ensure!(!self.imu_name.is_empty(), "missing IMU name");
        ensure!(
            self.startup_feedback_margin_radians.is_finite()
                && self.startup_feedback_margin_radians >= 0.,
            "invalid startup feedback margin"
        );
        for value in [
            self.connection_timeout_seconds,
            self.telemetry_timeout_seconds,
            self.max_motor_age_ms,
            self.max_imu_age_ms,
            self.max_group_age_ms,
            self.max_state_sample_span_ms,
            self.max_tilt_radians,
            self.max_angular_velocity,
            self.max_linear_acceleration,
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 86400.,
                "invalid hardware safety limit or timeout"
            );
        }
        Ok(())
    }
}

pub struct Hardware {
    pub robot: Robot,
    pub groups: Vec<MotorGroup>,
    pub config: Config,
    pub target_limits: crate::target_limits::TargetLimits,
    motors: Vec<String>,
}

impl Hardware {
    pub fn validate_policy_gains(&self, metadata: &PolicyMetadata) -> Result<()> {
        for group in &self.groups {
            for motor in group.motors() {
                let i = metadata
                    .coordinates
                    .iter()
                    .position(|name| name == motor.coordinate())
                    .context("hardware coordinate missing from policy")?;
                let limits = motor.limits();
                for (name, value, low, high) in [
                    ("kp", metadata.kp[i], limits.kp_min(), limits.kp_max()),
                    ("kd", metadata.kd[i], limits.kd_min(), limits.kd_max()),
                ] {
                    ensure!(
                        (low..=high).contains(&f64::from(value)),
                        "{} {name} exceeds hardware limits",
                        motor.coordinate()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn restrict_target_limits(
        &self,
        limits: &crate::target_limits::TargetLimits,
    ) -> Result<crate::target_limits::TargetLimits> {
        let mut limits = limits.clone();
        for (i, name) in self.motors.iter().enumerate() {
            let motor = self
                .robot
                .motors()
                .find(|motor| motor.name() == name)
                .context("configured motor missing from hardware inventory")?;
            let bounds = motor.limits();
            limits.restrict_motor(i, bounds.position_min(), bounds.position_max())?;
        }
        Ok(limits)
    }

    pub async fn connect(config: Config, runtime: &Locomotion) -> Result<Self> {
        config.validate()?;
        let duration = Duration::from_secs_f64(config.connection_timeout_seconds);
        let robot = tokio::time::timeout(
            duration,
            Robot::connect_with(ConnectOptions::new(&config.endpoint).connect_timeout(duration)),
        )
        .await
        .context("hardware connection timed out")??;
        let groups: Vec<_> = config
            .groups
            .iter()
            .map(|name| robot.group(name))
            .collect::<Result<_, _>>()?;
        let metadata = &runtime.policy.metadata;
        let mut by_coordinate = BTreeMap::new();
        for group in &groups {
            ensure!(
                group.initial_state() == GroupState::Idle,
                "group {} must start IDLE",
                group.name()
            );
            ensure!(
                group.control_rate_hz() + 1e-6 >= 1. / metadata.control_timestep,
                "group {} is slower than the policy",
                group.name()
            );
            for motor in group.motors() {
                ensure!(
                    motor
                        .initial_status()
                        .is_some_and(|status| status.state() == DeviceState::Online),
                    "motor {} is not online",
                    motor.name()
                );
                ensure!(
                    by_coordinate.insert(motor.coordinate(), motor).is_none(),
                    "duplicate hardware coordinate"
                );
            }
        }
        ensure!(
            by_coordinate.keys().copied().collect::<BTreeSet<_>>()
                == metadata.coordinates.iter().map(String::as_str).collect(),
            "hardware/policy coordinate mismatch"
        );
        let imu = robot
            .imus()
            .find(|imu| imu.name() == config.imu_name)
            .context("configured IMU is missing")?;
        ensure!(
            imu.initial_status()
                .is_some_and(|status| status.state() == DeviceState::Online),
            "IMU is not online"
        );
        ensure!(
            imu.output_frame() == runtime.imu_frame(),
            "hardware/policy IMU frame mismatch"
        );
        let mut motors = Vec::new();
        let mut target_limits = runtime.policy.target_limits.clone();
        for (i, coordinate) in metadata.coordinates.iter().enumerate() {
            let motor = by_coordinate[coordinate.as_str()];
            let limits = motor.limits();
            target_limits.restrict_motor(i, limits.position_min(), limits.position_max())?;
            ensure!(
                (limits.position_min()..=limits.position_max())
                    .contains(&f64::from(metadata.default_position[i])),
                "{coordinate} nominal position exceeds hardware limits"
            );
            motors.push(motor.name().to_owned());
        }
        let hardware = Self {
            robot,
            groups,
            config,
            target_limits,
            motors,
        };
        hardware.validate_policy_gains(metadata)?;
        Ok(hardware)
    }

    pub async fn feedback(&self, expected: proto::GroupState) -> Result<Feedback> {
        self.feedback_with_tilt(expected, Some(self.config.max_tilt_radians))
            .await
    }

    pub async fn feedback_with_tilt(
        &self,
        expected: proto::GroupState,
        max_tilt: Option<f64>,
    ) -> Result<Feedback> {
        let start = Instant::now();
        let deadline = Duration::from_secs_f64(self.config.telemetry_timeout_seconds);
        let mut client = self.robot.raw_client().grpc_client();
        let frame = loop {
            let remaining = deadline
                .checked_sub(start.elapsed())
                .context("hardware state timed out")?;
            let response = tokio::time::timeout(
                remaining,
                client.get_state_frame(proto::GetStateFrameRequest {
                    groups: self.config.groups.clone(),
                    imus: vec![self.config.imu_name.clone()],
                }),
            )
            .await
            .context("hardware state timed out")?;
            match response {
                Ok(response) => break response.into_inner(),
                Err(status) if status.code() == tonic::Code::FailedPrecondition => {
                    tokio::time::sleep(Duration::from_millis(10).min(remaining)).await;
                }
                Err(error) => return Err(error.into()),
            }
        };
        decode_feedback(
            &frame,
            &self.motors,
            &self.config,
            expected,
            start.elapsed(),
            max_tilt,
        )
    }
}

fn unique<'a, T>(
    values: &'a [T],
    name: impl Fn(&'a T) -> &'a str,
) -> Result<BTreeMap<&'a str, &'a T>> {
    let mut result = BTreeMap::new();
    for value in values {
        ensure!(
            result.insert(name(value), value).is_none(),
            "duplicate state entry"
        );
    }
    Ok(result)
}

fn decode_feedback(
    frame: &proto::GetStateFrameResponse,
    motor_names: &[String],
    config: &Config,
    expected: proto::GroupState,
    transport: Duration,
    max_tilt: Option<f64>,
) -> Result<Feedback> {
    let motors = unique(&frame.motors, |motor| &motor.motor_name)?;
    let groups = unique(&frame.groups, |group| &group.group)?;
    let imus = unique(&frame.imus, |imu| &imu.imu_name)?;
    ensure!(
        motors.keys().copied().collect::<BTreeSet<_>>()
            == motor_names.iter().map(String::as_str).collect()
            && groups.keys().copied().collect::<BTreeSet<_>>()
                == config.groups.iter().map(String::as_str).collect()
            && imus.len() == 1
            && imus.contains_key(config.imu_name.as_str()),
        "incomplete/unexpected hardware state"
    );
    let mut oldest = frame.captured_monotonic_ns;
    let mut newest = 0;
    let mut check_age = |stamp, age, limit| -> Result<()> {
        ensure!(
            frame.captured_monotonic_ns.checked_sub(stamp) == Some(age),
            "inconsistent hardware timestamp"
        );
        ensure!(
            age as f64 / 1e6 + transport.as_secs_f64() * 1000. <= limit,
            "stale hardware telemetry"
        );
        oldest = oldest.min(stamp);
        newest = newest.max(stamp);
        Ok(())
    };
    for motor in &frame.motors {
        check_age(
            motor.sample_monotonic_ns,
            motor.sample_age_ns,
            config.max_motor_age_ms,
        )?;
        ensure!(
            groups.contains_key(motor.group.as_str()) && motor.fault_bits == 0,
            "motor group/fault is unsafe"
        );
        ensure!(
            [
                motor.position,
                motor.velocity,
                motor.torque,
                motor.temperature
            ]
            .iter()
            .all(|x| x.is_finite()),
            "nonfinite motor feedback"
        );
    }
    for group in &frame.groups {
        check_age(
            group.sample_monotonic_ns,
            group.sample_age_ns,
            config.max_group_age_ms,
        )?;
        ensure!(
            group.state == expected as i32,
            "group {} is not {:?}: {}",
            group.group,
            expected,
            group.detail
        );
    }
    let imu = imus[config.imu_name.as_str()];
    check_age(
        imu.sample_monotonic_ns,
        imu.sample_age_ns,
        config.max_imu_age_ms,
    )?;
    ensure!(
        frame.oldest_sample_age_ns == frame.captured_monotonic_ns - oldest
            && frame.sample_span_ns == newest - oldest,
        "inconsistent state span"
    );
    ensure!(
        frame.sample_span_ns as f64 / 1e6 <= config.max_state_sample_span_ms,
        "hardware samples are too far apart"
    );
    let orientation = imu
        .orientation
        .as_ref()
        .context("missing IMU orientation")?;
    let orientation_xyzw = [orientation.x, orientation.y, orientation.z, orientation.w];
    let gravity = projected_gravity(orientation_xyzw)?;
    if let Some(max_tilt) = max_tilt {
        ensure!(
            (-gravity[2]).clamp(-1., 1.).acos() <= max_tilt,
            "robot tilt exceeds safety limit"
        );
    }
    let vector = |value: &Option<proto::Vector3>, limit: f64| -> Result<[f64; 3]> {
        let value = value.as_ref().context("missing IMU vector")?;
        let v = DVec3::new(value.x, value.y, value.z);
        ensure!(
            v.is_finite() && v.length() <= limit,
            "IMU vector exceeds safety limit"
        );
        Ok(v.to_array())
    };
    Ok(Feedback {
        arm_target: None,
        motor_position: motor_names
            .iter()
            .map(|name| motors[name.as_str()].position)
            .collect(),
        motor_velocity: motor_names
            .iter()
            .map(|name| motors[name.as_str()].velocity)
            .collect(),
        orientation_xyzw,
        angular_velocity: vector(&imu.angular_velocity, config.max_angular_velocity)?,
        linear_acceleration: vector(&imu.linear_acceleration, config.max_linear_acceleration)?,
        command: [0.; 3],
        torso_command: [0.; 3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Config, proto::GetStateFrameResponse, Vec<String>) {
        let config = Config {
            control: None,
            motion: None,
            endpoint: "http://localhost:1".into(),
            groups: vec!["limb".into()],
            imu_name: "sensor".into(),
            connection_timeout_seconds: 1.,
            telemetry_timeout_seconds: 1.,
            max_motor_age_ms: 50.,
            max_imu_age_ms: 50.,
            max_group_age_ms: 50.,
            max_state_sample_span_ms: 20.,
            startup_feedback_margin_radians: 0.,
            max_tilt_radians: 0.7,
            max_angular_velocity: 20.,
            max_linear_acceleration: 100.,
        };
        let frame = proto::GetStateFrameResponse {
            captured_monotonic_ns: 100_000_000,
            motors: vec![proto::StateMotor {
                motor_name: "motor".into(),
                group: "limb".into(),
                position: 0.2,
                sample_monotonic_ns: 90_000_000,
                sample_age_ns: 10_000_000,
                ..Default::default()
            }],
            groups: vec![proto::StateGroup {
                group: "limb".into(),
                state: proto::GroupState::Idle as i32,
                sample_monotonic_ns: 95_000_000,
                sample_age_ns: 5_000_000,
                ..Default::default()
            }],
            imus: vec![proto::StateImu {
                imu_name: "sensor".into(),
                orientation: Some(proto::Quaternion {
                    w: 1.,
                    ..Default::default()
                }),
                angular_velocity: Some(proto::Vector3::default()),
                linear_acceleration: Some(proto::Vector3::default()),
                sample_monotonic_ns: 98_000_000,
                sample_age_ns: 2_000_000,
                ..Default::default()
            }],
            oldest_sample_age_ns: 10_000_000,
            sample_span_ns: 8_000_000,
        };
        (config, frame, vec!["motor".into()])
    }

    #[test]
    fn snapshot_validation_includes_transport_delay() {
        let (config, frame, motors) = fixture();
        let decode = |latency| {
            decode_feedback(
                &frame,
                &motors,
                &config,
                proto::GroupState::Idle,
                Duration::from_millis(latency),
                Some(config.max_tilt_radians),
            )
        };
        assert_eq!(decode(1).unwrap().motor_position, vec![0.2]);
        assert!(decode(41).unwrap_err().to_string().contains("stale"));
    }

    #[test]
    fn motion_can_defer_absolute_tilt_but_not_feedback_safety() {
        let (config, mut frame, motors) = fixture();
        let q = glam::DQuat::from_rotation_y(1.4);
        frame.imus[0].orientation = Some(proto::Quaternion {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        });
        let decode = |frame: &proto::GetStateFrameResponse, limit| {
            decode_feedback(
                frame,
                &motors,
                &config,
                proto::GroupState::Idle,
                Duration::ZERO,
                limit,
            )
        };
        assert!(
            decode(&frame, Some(config.max_tilt_radians))
                .unwrap_err()
                .to_string()
                .contains("tilt")
        );
        assert!(decode(&frame, None).is_ok());
        let mut bad = frame.clone();
        bad.motors[0].fault_bits = 1;
        assert!(decode(&bad, None).is_err());
        let mut bad = frame.clone();
        bad.imus[0].orientation.as_mut().unwrap().w = f64::NAN;
        assert!(decode(&bad, None).is_err());
        let mut bad = frame.clone();
        bad.groups[0].state = proto::GroupState::Fault as i32;
        assert!(decode(&bad, None).is_err());
        let mut bad = frame.clone();
        bad.imus[0].angular_velocity.as_mut().unwrap().x = config.max_angular_velocity + 1.;
        assert!(decode(&bad, None).is_err());
        let mut bad = frame.clone();
        bad.imus[0].sample_age_ns = 1_000_000_000;
        assert!(decode(&bad, None).is_err());
    }

    #[test]
    fn incomplete_inconsistent_and_unsafe_frames_fail() {
        let (config, frame, motors) = fixture();
        let decode = |frame: &proto::GetStateFrameResponse| {
            decode_feedback(
                frame,
                &motors,
                &config,
                proto::GroupState::Idle,
                Duration::ZERO,
                Some(config.max_tilt_radians),
            )
        };
        let mut bad = frame.clone();
        bad.motors.push(bad.motors[0].clone());
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.imus.clear();
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.motors[0].sample_age_ns = 0;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.sample_span_ns = 0;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.groups[0].state = proto::GroupState::Fault as i32;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.motors[0].fault_bits = 1;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.motors[0].position = f64::NAN;
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.imus[0].orientation = Some(proto::Quaternion {
            x: 1.,
            ..Default::default()
        });
        assert!(decode(&bad).is_err());
        let mut bad = frame.clone();
        bad.imus[0].angular_velocity.as_mut().unwrap().x = 21.;
        assert!(decode(&bad).is_err());
        let mut bad = frame;
        bad.imus[0].linear_acceleration = None;
        assert!(decode(&bad).is_err());
    }
}
