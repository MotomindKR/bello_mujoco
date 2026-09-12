//! Stateful hardware observations from already-filtered hardwared telemetry.

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::{
    Policy,
    gait::{CommandConfig, GaitConfig, GaitState},
};

#[derive(Deserialize)]
struct ObservationConfig {
    scale: Scale,
}

#[derive(Deserialize)]
struct Scale {
    base_lin_acc: f32,
    base_ang_vel: f32,
    projected_gravity: f32,
    command: f32,
    gait_phase: f32,
    joint_pos: f32,
    joint_vel: f32,
    last_action: f32,
}

#[derive(Deserialize)]
struct Environment {
    observation: ObservationConfig,
    gait: GaitConfig,
}

#[derive(Deserialize)]
struct Settings {
    environment: Environment,
    command: CommandConfig,
    joint_from_motor: Vec<Vec<f64>>,
    default_joint_position: Vec<f64>,
    policy_from_hardware: [[f64; 3]; 3],
    left_axis: [f32; 2],
    action_clip: f32,
    hardware_imu_frame: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Feedback {
    pub motor_position: Vec<f64>,
    pub motor_velocity: Vec<f64>,
    pub orientation_xyzw: [f64; 4],
    pub angular_velocity: [f64; 3],
    /// Gravity-removed local acceleration, as published by hardwared.
    pub linear_acceleration: [f64; 3],
    pub command: [f32; 3],
}

#[derive(Serialize)]
pub struct Step {
    pub observation: Vec<f32>,
    pub raw_action: Vec<f32>,
    pub action: Vec<f32>,
    pub motor_target: Vec<f32>,
    pub gait: GaitState,
}

pub struct Locomotion {
    pub policy: Policy,
    settings: Settings,
    gait: GaitState,
    history: Vec<f32>,
    last_action: Vec<f32>,
}

impl Locomotion {
    pub fn new(policy: Policy) -> Result<Self> {
        ensure!(
            policy.metadata.task == "locomotion",
            "telemetry inference requires a locomotion bundle"
        );
        let settings: Settings =
            serde_json::from_value(policy.document.clone()).context("locomotion metadata")?;
        let n = policy.metadata.coordinates.len();
        ensure!(
            settings.joint_from_motor.len() == n
                && settings
                    .joint_from_motor
                    .iter()
                    .all(|r| r.len() == n && r.iter().all(|x| x.is_finite()))
                && settings.default_joint_position.len() == n
                && settings
                    .default_joint_position
                    .iter()
                    .all(|x| x.is_finite()),
            "invalid motor-to-joint mapping"
        );
        ensure!(
            settings.action_clip.is_finite() && settings.action_clip > 0.,
            "invalid action bound"
        );
        ensure!(
            !settings.hardware_imu_frame.is_empty(),
            "missing hardware IMU frame"
        );
        let r = settings.policy_from_hardware;
        let rotation = DMat3::from_cols_array_2d(&r).transpose();
        ensure!(
            (rotation * rotation.transpose()).abs_diff_eq(DMat3::IDENTITY, 1e-8),
            "invalid IMU rotation"
        );
        ensure!(
            (rotation.determinant() - 1.).abs() < 1e-8,
            "IMU rotation must be proper"
        );
        ensure!(
            (settings.left_axis[0].hypot(settings.left_axis[1]) - 1.).abs() < 1e-6,
            "invalid lateral axis"
        );
        settings
            .environment
            .gait
            .validate(&settings.command, policy.metadata.control_timestep as f32)?;
        let scale = &settings.environment.observation.scale;
        for (name, size) in &policy.metadata.observation_frame {
            let (expected, factor) = match name.as_str() {
                "imu_linear_acceleration" => (3, scale.base_lin_acc),
                "imu_angular_velocity" => (3, scale.base_ang_vel),
                "projected_gravity" => (3, scale.projected_gravity),
                "command" => (3, scale.command),
                "gait_phase" => (3, scale.gait_phase),
                "joint_position" => (policy.metadata.policy_indices.len(), scale.joint_pos),
                "joint_velocity" => (policy.metadata.policy_indices.len(), scale.joint_vel),
                "last_action" => (policy.metadata.policy_indices.len(), scale.last_action),
                _ => bail!("unsupported hardware observation {name}"),
            };
            ensure!(
                *size == expected && factor.is_finite(),
                "invalid observation component {name}"
            );
        }
        let last_action = vec![0.; policy.metadata.policy_indices.len()];
        Ok(Self {
            policy,
            settings,
            gait: GaitState::default(),
            history: Vec::new(),
            last_action,
        })
    }

    pub fn imu_frame(&self) -> &str {
        &self.settings.hardware_imu_frame
    }

    pub fn reset(&mut self) {
        self.gait = GaitState::default();
        self.history.clear();
        self.last_action.fill(0.);
    }

    pub fn nominal_tolerance(&self, index: usize, direct: f64, coupled: f64) -> f64 {
        if self
            .settings
            .joint_from_motor
            .iter()
            .filter(|row| row[index].abs() > 1e-12)
            .count()
            > 1
        {
            coupled
        } else {
            direct
        }
    }

    pub fn remote_command(&self, axes: [f32; 3], scale: f32) -> Result<[f32; 3]> {
        ensure!(
            axes.iter()
                .all(|v| v.is_finite() && (-1. ..=1.).contains(v))
                && scale.is_finite()
                && scale > 0.
                && scale <= 1.,
            "invalid joystick intent/scale"
        );
        let c = &self.settings.command;
        if !c.enabled {
            return Ok([0.; 3]);
        }
        let [lx, ly] = self.settings.left_axis;
        let normalized = [
            axes[0] * ly - axes[1] * lx,
            -axes[0] * lx - axes[1] * ly,
            axes[2],
        ];
        let mut output = [0.; 3];
        for (i, (bounds, minimum)) in [c.lin_vel_x, c.lin_vel_y, c.ang_vel_yaw]
            .into_iter()
            .zip([
                c.minimum_moving_linear_speed,
                c.minimum_moving_linear_speed,
                c.minimum_moving_yaw_speed,
            ])
            .enumerate()
        {
            ensure!(
                minimum.is_finite() && minimum >= 0.,
                "invalid minimum command speed"
            );
            let [low, high] = bounds.map(|v| v * scale);
            ensure!(
                !(bounds[0] <= -minimum && low > -minimum)
                    && !(bounds[1] >= minimum && high < minimum),
                "joystick scale removes a trained moving range"
            );
            let value = normalized[i].clamp(-1., 1.);
            output[i] = if value > 0. && high >= minimum {
                minimum + value * (high - minimum)
            } else if value < 0. && low <= -minimum {
                -minimum - value * (low + minimum)
            } else {
                0.
            };
        }
        Ok(output)
    }

    pub fn step(&mut self, sample: &Feedback) -> Result<Step> {
        let n = self.policy.metadata.coordinates.len();
        ensure!(
            sample.motor_position.len() == n && sample.motor_velocity.len() == n,
            "motor feedback shape mismatch"
        );
        ensure!(
            sample
                .motor_position
                .iter()
                .chain(&sample.motor_velocity)
                .chain(&sample.angular_velocity)
                .chain(&sample.linear_acceleration)
                .all(|x| x.is_finite()),
            "nonfinite motor/IMU feedback"
        );
        let gravity = projected_gravity(sample.orientation_xyzw)?;
        let rotation = DMat3::from_cols_array_2d(&self.settings.policy_from_hardware).transpose();
        let rotate = |v| (rotation * DVec3::from_array(v)).to_array();
        let gravity = rotate(gravity);
        let acceleration = rotate(sample.linear_acceleration);
        let angular_velocity = rotate(sample.angular_velocity);
        let force: [f32; 3] =
            std::array::from_fn(|i| acceleration[i] as f32 - 9.80665_f32 * gravity[i] as f32);
        let s = &self.settings;
        self.gait.advance(
            sample.command,
            &s.environment.gait,
            &s.command,
            s.left_axis,
            self.policy.metadata.control_timestep as f32,
        )?;
        let gait = self.gait.features(&s.environment.gait, &s.command);
        let scale = &s.environment.observation.scale;
        let mut observation = Vec::new();
        for (name, _) in &self.policy.metadata.observation_frame {
            let values: Vec<f32> = match name.as_str() {
                "imu_linear_acceleration" => force.map(|x| x * scale.base_lin_acc).to_vec(),
                "imu_angular_velocity" => angular_velocity
                    .map(|x| (x * scale.base_ang_vel as f64) as f32)
                    .to_vec(),
                "projected_gravity" => gravity.map(|x| x as f32 * scale.projected_gravity).to_vec(),
                "command" => self.gait.command.map(|x| x * scale.command).to_vec(),
                "gait_phase" => gait.map(|x| x * scale.gait_phase).to_vec(),
                "last_action" => self
                    .last_action
                    .iter()
                    .map(|x| x * scale.last_action)
                    .collect(),
                "joint_position" | "joint_velocity" => {
                    let position = name == "joint_position";
                    let motors = if position {
                        &sample.motor_position
                    } else {
                        &sample.motor_velocity
                    };
                    self.policy
                        .metadata
                        .policy_indices
                        .iter()
                        .map(|&i| {
                            let value: f64 = s.joint_from_motor[i]
                                .iter()
                                .zip(motors)
                                .map(|(a, b)| a * b)
                                .sum();
                            ((value
                                - if position {
                                    s.default_joint_position[i]
                                } else {
                                    0.
                                })
                                * if position {
                                    scale.joint_pos as f64
                                } else {
                                    scale.joint_vel as f64
                                }) as f32
                        })
                        .collect()
                }
                _ => unreachable!("validated observation component"),
            };
            observation.extend(values);
        }
        if self.history.is_empty() {
            self.history = observation.repeat(self.policy.metadata.observation_stack_size);
        } else {
            self.history.rotate_left(observation.len());
            let start = self.history.len() - observation.len();
            self.history[start..].copy_from_slice(&observation);
        }
        let output = self.policy.infer(&self.history)?;
        self.last_action = output
            .action
            .iter()
            .map(|a| a.clamp(-s.action_clip, s.action_clip))
            .collect();
        Ok(Step {
            observation,
            raw_action: output.action,
            action: self.last_action.clone(),
            motor_target: output.motor_target,
            gait: self.gait.clone(),
        })
    }
}

pub fn projected_gravity(q: [f64; 4]) -> Result<[f64; 3]> {
    let q = DQuat::from_array(q);
    ensure!(
        q.is_finite() && (0.9..=1.1).contains(&q.length()),
        "invalid IMU quaternion"
    );
    Ok((q.normalize().conjugate() * DVec3::NEG_Z).to_array())
}
