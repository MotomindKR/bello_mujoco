//! SONIC reference/proprioception assembly for prerecorded and live references.

use anyhow::{Context, Result, ensure};
use glam::{DQuat, DVec3, EulerRot};
use mujoco_rs::prelude::{MjModel, MjtJoint, MjtObj};
use serde::{Deserialize, Serialize};

use crate::Policy;
use crate::motion::clip::Clip;

#[derive(Deserialize)]
struct Settings {
    #[serde(default)]
    smpl_enabled: bool,
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
    /// Logical joints in `Sonic::joint_names()` order, for both live and clip input.
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub orientation: DQuat,
    pub smpl: Option<SmplFrame>,
}

#[derive(Clone, Debug)]
pub struct SmplFrame {
    pub joints: Vec<DVec3>,
    pub orientation: DQuat,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceInput {
    #[default]
    Robot,
    Smpl,
}

struct ReferenceLayout {
    qpos_indices: Vec<usize>,
    qvel_indices: Vec<usize>,
}

impl ReferenceLayout {
    fn load(physics: &[u8], joint_names: &[String]) -> Result<Self> {
        ensure!(
            unsafe { mujoco_rs::mujoco_c::mj_version() }
                == mujoco_rs::mujoco_c::mjVERSION_HEADER as i32,
            "loaded MuJoCo must match the pinned bindings"
        );
        let model = MjModel::from_xml_string(std::str::from_utf8(physics)?)?;
        let n = joint_names.len();
        ensure!(
            model.nq() as usize == n + 7
                && model.nv() as usize == n + 6
                && model.jnt_type()[0] == MjtJoint::mjJNT_FREE,
            "SONIC model must have one floating base and scalar joints"
        );
        let mut qpos_indices = Vec::with_capacity(n);
        let mut qvel_indices = Vec::with_capacity(n);
        for name in joint_names {
            let id = model
                .name_to_id(MjtObj::mjOBJ_JOINT, name)
                .with_context(|| format!("missing SONIC joint {name}"))?;
            ensure!(
                matches!(
                    model.jnt_type()[id],
                    MjtJoint::mjJNT_HINGE | MjtJoint::mjJNT_SLIDE
                ),
                "SONIC joint must be scalar: {name}"
            );
            let qpos = model.jnt_qposadr()[id] as usize;
            let qvel = model.jnt_dofadr()[id] as usize;
            ensure!(
                qpos >= 7 && qvel >= 6 && qpos - 7 == qvel - 6,
                "invalid SONIC reference coordinates for {name}"
            );
            ensure!(
                !qpos_indices.contains(&qpos),
                "duplicate SONIC joint {name}"
            );
            qpos_indices.push(qpos);
            qvel_indices.push(qvel);
        }
        Ok(Self {
            qpos_indices,
            qvel_indices,
        })
    }
}

impl Settings {
    fn encode_joints(&self, frame: &ReferenceFrame) -> (Vec<f64>, Vec<f64>) {
        let mut position = self.source_neutral.clone();
        let mut velocity = vec![0.; position.len()];
        for (target, &source) in self.source_joint_indices.iter().enumerate() {
            let sign = self.source_signs[target];
            position[source] = self.source_neutral[source]
                + (frame.position[target] - self.default_joint_position[target]) * sign;
            velocity[source] = frame.velocity[target] * sign;
        }
        (position, velocity)
    }
}

pub struct Sonic {
    pub policy: Policy,
    settings: Settings,
    clip: Option<Clip>,
    alignment: Option<DQuat>,
    history: Vec<f32>,
    last_action: Vec<f32>,
    root_from_hardware_imu: DQuat,
    reference_layout: ReferenceLayout,
    input: ReferenceInput,
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
        let reference_layout = ReferenceLayout::load(&policy.physics_xml, &settings.joint_names)?;
        let root_from_hardware_imu =
            checked_quaternion(DQuat::from_array(settings.root_from_hardware_imu_xyzw))?;
        let mut layout = vec![("sonic_future_reference", settings.future_frames * 64)];
        if settings.smpl_enabled {
            layout.extend([("sonic_smpl_reference", 840), ("sonic_smpl_selected", 1)]);
        }
        layout.extend([
            ("gravity_direction", 3),
            ("base_angular_velocity", 3),
            ("joint_position", n),
            ("joint_velocity", n),
            ("last_action", n),
        ]);
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
            input: ReferenceInput::Robot,
            policy,
            settings,
            clip,
            alignment: None,
            history: Vec::new(),
            last_action: vec![0.; n],
            root_from_hardware_imu,
            reference_layout,
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
        if self.input == ReferenceInput::Smpl {
            0.02
        } else {
            self.settings.future_dt
        }
    }

    pub fn select_input(&mut self, input: ReferenceInput) -> Result<()> {
        ensure!(
            input != ReferenceInput::Smpl || self.settings.smpl_enabled,
            "SONIC bundle has no SMPL encoder"
        );
        self.input = input;
        Ok(())
    }
    pub fn root_from_imu(&self) -> DQuat {
        self.root_from_hardware_imu
    }

    pub fn joint_positions(&self, motor_position: &[f64]) -> Vec<f64> {
        self.settings
            .joint_from_motor
            .iter()
            .map(|row| row.iter().zip(motor_position).map(|(a, b)| a * b).sum())
            .collect()
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
                    smpl: None,
                    position: self
                        .reference_layout
                        .qpos_indices
                        .iter()
                        .map(|&i| sample.qpos[i])
                        .collect(),
                    velocity: self
                        .reference_layout
                        .qvel_indices
                        .iter()
                        .map(|&i| sample.qvel[i])
                        .collect(),
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
        self.step_reference_input(feedback, alignment_root, references, self.input)
    }

    pub fn step_reference_input(
        &mut self,
        feedback: &Feedback,
        alignment_root: DQuat,
        references: &[ReferenceFrame],
        input: ReferenceInput,
    ) -> Result<Step> {
        ensure!(
            input != ReferenceInput::Smpl || self.settings.smpl_enabled,
            "SONIC bundle has no SMPL encoder"
        );
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
            let (source_position, source_velocity) = self.settings.encode_joints(sample);
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
        if self.settings.smpl_enabled {
            if input == ReferenceInput::Smpl {
                for sample in references {
                    let human = sample.smpl.as_ref().context("SMPL reference missing")?;
                    ensure!(
                        human.joints.len() == 24
                            && human.joints.iter().all(|v| v.is_finite())
                            && human.joints[0].length() < 1e-5
                            && human.joints[15].length() > 0.1,
                        "invalid SMPL human geometry"
                    );
                    let root = checked_quaternion(human.orientation)?;
                    for joint in &human.joints {
                        observation.extend((root.conjugate() * *joint).to_array());
                    }
                    observation.extend(rotation_6d(orientation.conjugate() * alignment * root));
                    let (joints, _) = self.settings.encode_joints(sample);
                    observation.extend(&joints[23..29]);
                }
                observation.push(1.);
            } else {
                observation.extend([0.; 841]);
            }
        }
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

    const PHYSICS: &[u8] = br#"<mujoco><worldbody><body><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body><joint name="a"/><geom type="sphere" size=".1" mass="1"/></body>
        <body><joint name="b"/><geom type="sphere" size=".1" mass="1"/></body>
        <body><joint name="c"/><geom type="sphere" size=".1" mass="1"/></body>
        </body></worldbody></mujoco>"#;

    #[test]
    fn reference_encoding_uses_named_joint_order() {
        let names = ["c", "a", "b"].map(String::from);
        let layout = ReferenceLayout::load(PHYSICS, &names).unwrap();
        assert_eq!(layout.qpos_indices, [9, 7, 8]);
        assert_eq!(layout.qvel_indices, [8, 6, 7]);
        let settings = Settings {
            smpl_enabled: false,
            hardware_imu_frame: "imu".into(),
            joint_names: names.to_vec(),
            joint_from_motor: vec![],
            default_joint_position: vec![0.1, 0.2, 0.3],
            action_clip: 20.,
            reference_clips: vec![],
            future_frames: 10,
            future_dt: 0.1,
            source_joint_indices: vec![2, 0, 3],
            source_neutral: vec![10., 20., 30., 40.],
            source_signs: vec![-1., 1., -1.],
            root_from_hardware_imu_xyzw: [0., 0., 0., 1.],
        };
        let live = ReferenceFrame {
            smpl: None,
            position: vec![3., 1., 2.],
            velocity: vec![30., 10., 20.],
            orientation: DQuat::IDENTITY,
        };
        let (position, velocity) = settings.encode_joints(&live);
        assert_eq!(position, [10.8, 20., 27.1, 38.3]);
        assert_eq!(velocity, [10., 0., -30., -20.]);
        // Clip storage order must not change the encoder's named joint inputs.
        let qpos = [0., 0., 1., 1., 0., 0., 0., 1., 2., 3.];
        let qvel = [0., 0., 0., 0., 0., 0., 10., 20., 30.];
        let clip = ReferenceFrame {
            smpl: None,
            position: layout.qpos_indices.iter().map(|&i| qpos[i]).collect(),
            velocity: layout.qvel_indices.iter().map(|&i| qvel[i]).collect(),
            orientation: DQuat::IDENTITY,
        };
        assert_eq!(settings.encode_joints(&clip), (position, velocity));
        assert_eq!(live.position, [3., 1., 2.]);
    }

    #[test]
    fn reference_layout_rejects_missing_duplicate_and_nonfloating_models() {
        for names in [["a", "b", "missing"], ["a", "b", "a"], ["a", "b", "c"]] {
            let physics = if names == ["a", "b", "c"] {
                b"<mujoco/>".as_slice()
            } else {
                PHYSICS
            };
            assert!(ReferenceLayout::load(physics, &names.map(String::from)).is_err());
        }
    }

    #[test]
    fn rotation_6d_is_row_major_first_two_columns() {
        let value = rotation_6d(DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2));
        let expected = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0];
        for (actual, expected) in value.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
}
