//! SONIC reference/proprioception assembly from bundled prerecorded clips.

use anyhow::{Context, Result, ensure};
use glam::{DQuat, DVec3, EulerRot};
use serde::{Deserialize, Serialize};

use crate::Policy;
use crate::motion::clip::Clip;

#[derive(Deserialize)]
struct Settings {
    hardware_imu_frame: String,
    joint_names: Vec<String>,
    joint_from_motor: Vec<Vec<f64>>,
    default_joint_position: Vec<f64>,
    action_clip: f32,
    reference_clips: Vec<String>,
    future_frames: usize,
    future_dt: f64,
    source_joint_indices: Vec<usize>,
    source_neutral: Vec<f64>,
    source_signs: Vec<f64>,
    root_from_hardware_imu_xyzw: [f64; 4],
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Feedback {
    pub motor_position: Vec<f64>,
    pub motor_velocity: Vec<f64>,
    pub orientation_xyzw: [f64; 4],
    pub angular_velocity_local: [f64; 3],
    pub reference_time: f64,
    pub initialize: bool,
}

#[derive(Serialize)]
pub struct Step {
    pub observation: Vec<f32>,
    pub raw_action: Vec<f32>,
    pub action: Vec<f32>,
    pub motor_target: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct ReferenceFrame {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub orientation: DQuat,
}

pub struct Sonic {
    pub policy: Policy,
    settings: Settings,
    clip: Option<Clip>,
    alignment: Option<DQuat>,
    history: Vec<f32>,
    last_action: Vec<f32>,
    root_from_hardware_imu: DQuat,
}

impl Sonic {
    pub fn new(policy: Policy, clip_name: &str) -> Result<Self> {
        Self::load(policy, Some(clip_name))
    }

    pub fn live(policy: Policy) -> Result<Self> {
        Self::load(policy, None)
    }

    fn load(mut policy: Policy, clip_name: Option<&str>) -> Result<Self> {
        ensure!(
            policy.metadata.task == "sonic_transfer",
            "SONIC bundle required"
        );
        let settings: Settings = serde_json::from_value(policy.document.clone())?;
        let clip = clip_name
            .map(|name| {
                ensure!(
                    settings.reference_clips.iter().any(|clip| clip == name),
                    "clip {name:?} is not selected in this bundle"
                );
                Clip::load(
                    &policy
                        .reference_clips
                        .remove(name)
                        .with_context(|| format!("bundle has no clip {name:?}"))?,
                )
            })
            .transpose()?;
        let n = policy.metadata.policy_indices.len();
        ensure!(
            n == 28
                && clip
                    .as_ref()
                    .is_none_or(|clip| clip.joint_names == settings.joint_names)
                && settings.joint_names.len() == n
                && settings.joint_from_motor.len() == n
                && settings
                    .joint_from_motor
                    .iter()
                    .all(|row| row.len() == n && row.iter().all(|v| v.is_finite()))
                && settings.default_joint_position.len() == n
                && settings.source_joint_indices.len() == n
                && settings.source_signs.len() == n
                && settings.source_neutral.len() == n + 1,
            "invalid SONIC joint mapping"
        );
        ensure!(
            settings
                .source_joint_indices
                .iter()
                .all(|&index| index < settings.source_neutral.len())
                && settings
                    .source_joint_indices
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == n
                && settings
                    .source_signs
                    .iter()
                    .all(|value| value.is_finite() && value.abs() == 1.)
                && settings
                    .source_neutral
                    .iter()
                    .all(|value| value.is_finite()),
            "invalid SONIC source codec"
        );
        ensure!(
            settings.future_frames == 10
                && settings.future_dt.is_finite()
                && settings.future_dt > 0.
                && settings.action_clip.is_finite()
                && settings.action_clip > 0.,
            "invalid SONIC timing or action settings"
        );
        ensure!(
            !settings.hardware_imu_frame.is_empty(),
            "missing SONIC hardware IMU frame"
        );
        let root_from_hardware_imu =
            checked_quaternion(DQuat::from_array(settings.root_from_hardware_imu_xyzw))?;
        let layout = [
            ("sonic_future_reference", settings.future_frames * 64),
            ("gravity_direction", 3),
            ("base_angular_velocity", 3),
            ("joint_position", n),
            ("joint_velocity", n),
            ("last_action", n),
        ];
        ensure!(
            policy
                .metadata
                .observation_frame
                .iter()
                .map(|(name, width)| (name.as_str(), *width))
                .eq(layout),
            "unsupported SONIC observation layout"
        );
        Ok(Self {
            policy,
            settings,
            clip,
            alignment: None,
            history: Vec::new(),
            last_action: vec![0.; n],
            root_from_hardware_imu,
        })
    }

    pub fn clip_names(policy: &Policy) -> Result<Vec<String>> {
        ensure!(
            policy.metadata.task == "sonic_transfer",
            "SONIC bundle required"
        );
        let settings: Settings = serde_json::from_value(policy.document.clone())?;
        Ok(settings.reference_clips)
    }

    pub fn reset(&mut self) {
        self.alignment = None;
        self.history.clear();
        self.last_action.fill(0.);
    }

    pub fn imu_frame(&self) -> &str {
        &self.settings.hardware_imu_frame
    }

    pub fn joint_names(&self) -> &[String] {
        &self.settings.joint_names
    }
    pub fn future_dt(&self) -> f64 {
        self.settings.future_dt
    }
    pub fn root_from_imu(&self) -> DQuat {
        self.root_from_hardware_imu
    }

    pub fn step(&mut self, feedback: &Feedback) -> Result<Step> {
        let clip = self
            .clip
            .as_ref()
            .context("SONIC playback requires a clip")?;
        ensure!(
            feedback.reference_time.is_finite()
                && feedback.reference_time >= 0.
                && feedback.reference_time < clip.duration(),
            "invalid SONIC reference time"
        );
        let fps = clip.fps() as f32;
        let initial = clip.sample_frame(f64::from(feedback.reference_time as f32 * fps));
        let alignment = from_wxyz(initial.qpos[3..7].try_into()?);
        let frame =
            (feedback.reference_time as f32 + self.policy.metadata.control_timestep as f32) * fps;
        let references = (0..self.settings.future_frames)
            .map(|i| {
                let offset = (i as f32 * self.settings.future_dt as f32 * fps).round_ties_even();
                let sample = clip.sample_frame(f64::from(frame + offset));
                ReferenceFrame {
                    position: sample.qpos[7..].to_vec(),
                    velocity: sample.qvel[6..].to_vec(),
                    orientation: from_wxyz(sample.qpos[3..7].try_into().unwrap()),
                }
            })
            .collect::<Vec<_>>();
        self.step_reference(feedback, alignment, &references)
    }

    pub fn step_reference(
        &mut self,
        feedback: &Feedback,
        alignment_root: DQuat,
        references: &[ReferenceFrame],
    ) -> Result<Step> {
        let n = self.settings.joint_names.len();
        ensure!(
            feedback.motor_position.len() == n && feedback.motor_velocity.len() == n,
            "SONIC motor feedback shape mismatch"
        );
        ensure!(
            feedback
                .motor_position
                .iter()
                .chain(&feedback.motor_velocity)
                .chain(&feedback.orientation_xyzw)
                .chain(&feedback.angular_velocity_local)
                .all(|value| value.is_finite()),
            "invalid SONIC feedback"
        );
        let hardware_orientation =
            checked_quaternion(DQuat::from_array(feedback.orientation_xyzw))?;
        let orientation = hardware_orientation * self.root_from_hardware_imu.conjugate();
        if feedback.initialize {
            self.reset();
            let reference_root = checked_quaternion(alignment_root)?;
            self.alignment = Some(DQuat::from_rotation_z(
                orientation.to_euler(EulerRot::ZYX).0 - reference_root.to_euler(EulerRot::ZYX).0,
            ));
        }
        let alignment = self
            .alignment
            .context("initialize SONIC clip before inference")?;
        let map = |values: &[f64]| -> Vec<f64> {
            self.settings
                .joint_from_motor
                .iter()
                .map(|row| row.iter().zip(values).map(|(a, b)| a * b).sum())
                .collect()
        };
        let position = map(&feedback.motor_position);
        let velocity = map(&feedback.motor_velocity);
        let mut source_positions =
            Vec::with_capacity(self.settings.future_frames * self.settings.source_neutral.len());
        let mut source_velocities =
            Vec::with_capacity(self.settings.future_frames * self.settings.source_neutral.len());
        let mut root_orientations = Vec::with_capacity(self.settings.future_frames);
        ensure!(
            references.len() == self.settings.future_frames,
            "invalid SONIC reference window"
        );
        for sample in references {
            ensure!(
                sample.position.len() == n
                    && sample.velocity.len() == n
                    && sample
                        .position
                        .iter()
                        .chain(&sample.velocity)
                        .all(|v| v.is_finite()),
                "invalid SONIC reference joints"
            );
            let reference_orientation = checked_quaternion(sample.orientation)?;
            let mut source_position = self.settings.source_neutral.clone();
            let mut source_velocity = vec![0.; source_position.len()];
            for target in 0..n {
                let source = self.settings.source_joint_indices[target];
                let sign = self.settings.source_signs[target];
                source_position[source] = self.settings.source_neutral[source]
                    + (sample.position[target] - self.settings.default_joint_position[target])
                        * sign;
                source_velocity[source] = sample.velocity[target] * sign;
            }
            source_positions.extend(source_position);
            source_velocities.extend(source_velocity);
            root_orientations.push(rotation_6d(
                orientation.conjugate() * alignment * reference_orientation,
            ));
        }
        // Preserve the pretrained release's historical non-flat reshape: the
        // complete position block precedes the complete velocity block, that
        // buffer is reshaped into 58-wide rows, and orientations are appended
        // row-by-row.  These rows are not individual reference time steps.
        let mut joint_command = source_positions;
        joint_command.extend(source_velocities);
        let command_width = 2 * self.settings.source_neutral.len();
        let mut reference = Vec::with_capacity(self.settings.future_frames * 64);
        for (command, root_orientation) in joint_command
            .chunks_exact(command_width)
            .zip(root_orientations)
        {
            reference.extend(command);
            reference.extend(root_orientation);
        }
        let gravity = orientation.conjugate() * DVec3::NEG_Z;
        let mut observation = reference;
        observation.extend(gravity.to_array());
        observation.extend(
            (self.root_from_hardware_imu * DVec3::from_array(feedback.angular_velocity_local))
                .to_array(),
        );
        observation.extend(
            position
                .iter()
                .zip(&self.settings.default_joint_position)
                .map(|(value, default)| value - default),
        );
        observation.extend(velocity);
        observation.extend(self.last_action.iter().map(|value| f64::from(*value)));
        let frame: Vec<f32> = observation.into_iter().map(|value| value as f32).collect();
        if self.history.is_empty() {
            self.history = frame.repeat(self.policy.metadata.observation_stack_size);
        } else {
            self.history.rotate_left(frame.len());
            let start = self.history.len() - frame.len();
            self.history[start..].copy_from_slice(&frame);
        }
        let output = self.policy.infer(&self.history)?;
        self.last_action = output
            .action
            .iter()
            .map(|value| value.clamp(-self.settings.action_clip, self.settings.action_clip))
            .collect();
        Ok(Step {
            observation: self.history.clone(),
            raw_action: output.action,
            action: self.last_action.clone(),
            motor_target: output.motor_target,
        })
    }
}

fn checked_quaternion(value: DQuat) -> Result<DQuat> {
    ensure!(
        value.is_finite() && (0.9..=1.1).contains(&value.length()),
        "invalid SONIC orientation"
    );
    Ok(value.normalize())
}

fn from_wxyz(value: [f64; 4]) -> DQuat {
    DQuat::from_xyzw(value[1], value[2], value[3], value[0])
}

fn rotation_6d(value: DQuat) -> [f64; 6] {
    let matrix = glam::DMat3::from_quat(value);
    [
        matrix.x_axis.x,
        matrix.y_axis.x,
        matrix.x_axis.y,
        matrix.y_axis.y,
        matrix.x_axis.z,
        matrix.y_axis.z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_6d_is_row_major_first_two_columns() {
        let value = rotation_6d(DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2));
        let expected = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0];
        for (actual, expected) in value.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
}
