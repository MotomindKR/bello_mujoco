//! Mini-native SuperMimic history and reference codec for prerecorded playback.

use anyhow::{Context, Result, ensure};
use glam::{DMat3, DQuat, DVec3};
use mujoco_rs::prelude::{MjModel, MjtJoint, MjtObj};
use serde::{Deserialize, Serialize};

use crate::Policy;
use crate::motion::clip::{Clip, Sample};

#[derive(Deserialize)]
struct Settings {
    joint_names: Vec<String>,
    joint_from_motor: Vec<Vec<f64>>,
    default_joint_position: Vec<f64>,
    collection_action_clip: f32,
    simulation_control: String,
    reference_clips: Vec<String>,
    reference_root_body: String,
    hardware_imu_frame: String,
    root_from_hardware_imu_xyzw: [f64; 4],
    sample_frame: Vec<(String, usize)>,
    history_frames: usize,
    future_frames: usize,
    reference_orientation: bool,
    goal_joint_velocity_scale: f64,
    rotation_6d_layout: String,
}

#[derive(Deserialize)]
pub struct Feedback {
    pub motor_position: Vec<f64>,
    pub motor_velocity: Vec<f64>,
    pub previous_motor_target: Vec<f64>,
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
    pub requested_motor_target: Vec<f64>,
    pub motor_target: Vec<f64>,
    pub limited: bool,
}

struct Layout {
    qpos: Vec<usize>,
    qvel: Vec<usize>,
    root_qpos: usize,
    root_qvel: usize,
    root_offset: DVec3,
    root_rotation: DQuat,
}

fn from_wxyz(q: [f64; 4]) -> DQuat {
    DQuat::from_xyzw(q[1], q[2], q[3], q[0])
}

fn checked(q: DQuat) -> Result<DQuat> {
    ensure!(
        q.is_finite() && (0.9..=1.1).contains(&q.length()),
        "invalid SuperMimic orientation"
    );
    Ok(q.normalize())
}

fn rotation_6d(q: DQuat) -> [f64; 6] {
    let matrix = DMat3::from_quat(q.normalize());
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
    fn rotation_encoding_matches_reference_row_major_columns() {
        let values = rotation_6d(DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2));
        for (actual, expected) in values.into_iter().zip([0., -1., 1., 0., 0., 0.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

fn heading(q: DQuat) -> DQuat {
    let forward = q * DVec3::X;
    let yaw = if forward.x * forward.x + forward.y * forward.y >= 1e-12 {
        forward.y.atan2(forward.x)
    } else {
        0.0
    };
    DQuat::from_rotation_z(yaw)
}

fn fixed_offset(model: &MjModel, root: usize, frame: usize) -> Result<(DVec3, DQuat)> {
    let mut path = Vec::new();
    let mut body = frame;
    while body != root {
        ensure!(
            body != 0 && model.body_jntnum()[body] == 0,
            "SuperMimic frame must be fixed to its root"
        );
        path.push(body);
        body = model.body_parentid()[body] as usize;
    }
    let mut position = DVec3::ZERO;
    let mut quaternion = DQuat::IDENTITY;
    for body in path.into_iter().rev() {
        position += quaternion * DVec3::from_array(model.body_pos()[body]);
        quaternion *= from_wxyz(model.body_quat()[body]);
    }
    Ok((position, quaternion.normalize()))
}

impl Layout {
    fn load(physics: &[u8], settings: &Settings, clip: &Clip) -> Result<Self> {
        ensure!(
            unsafe { mujoco_rs::mujoco_c::mj_version() }
                == mujoco_rs::mujoco_c::mjVERSION_HEADER as i32,
            "loaded MuJoCo must match the pinned bindings"
        );
        let model = MjModel::from_xml_string(std::str::from_utf8(physics)?)?;
        let n = settings.joint_names.len();
        ensure!(
            model.nq() as usize == n + 7
                && model.nv() as usize == n + 6
                && clip.nq == n + 7
                && clip.nv == n + 6,
            "invalid SuperMimic physics coordinates"
        );
        let root = model
            .name_to_id(MjtObj::mjOBJ_BODY, &settings.reference_root_body)
            .context("missing SuperMimic reference root")?;
        let mut free_body = root;
        while free_body != 0 && model.body_jntnum()[free_body] == 0 {
            free_body = model.body_parentid()[free_body] as usize;
        }
        ensure!(
            free_body != 0
                && model.body_parentid()[free_body] == 0
                && model.body_jntnum()[free_body] == 1,
            "SuperMimic requires one free root"
        );
        let free = model.body_jntadr()[free_body] as usize;
        ensure!(
            model.jnt_type()[free] == MjtJoint::mjJNT_FREE,
            "SuperMimic root must be free"
        );
        let (root_offset, root_rotation) = fixed_offset(&model, free_body, root)?;
        let imu = model
            .name_to_id(MjtObj::mjOBJ_BODY, &settings.hardware_imu_frame)
            .context("missing SuperMimic hardware IMU frame")?;
        let (_, expected_imu) = fixed_offset(&model, root, imu)?;
        ensure!(
            expected_imu.angle_between(checked(DQuat::from_array(
                settings.root_from_hardware_imu_xyzw
            ))?) < 1e-5,
            "SuperMimic IMU transform differs from bundled physics"
        );
        let mut qpos = Vec::with_capacity(n);
        let mut qvel = Vec::with_capacity(n);
        for name in &settings.joint_names {
            let joint = model
                .name_to_id(MjtObj::mjOBJ_JOINT, name)
                .with_context(|| format!("missing SuperMimic joint {name}"))?;
            ensure!(
                matches!(
                    model.jnt_type()[joint],
                    MjtJoint::mjJNT_HINGE | MjtJoint::mjJNT_SLIDE
                ),
                "SuperMimic joints must be scalar"
            );
            let mut body = model.jnt_bodyid()[joint] as usize;
            while body != free_body && body != 0 {
                body = model.body_parentid()[body] as usize;
            }
            ensure!(
                body == free_body,
                "SuperMimic joint is outside the root tree"
            );
            qpos.push(model.jnt_qposadr()[joint] as usize);
            qvel.push(model.jnt_dofadr()[joint] as usize);
        }
        ensure!(
            clip.free_qpos == [model.jnt_qposadr()[free] as usize],
            "SuperMimic clip free root differs from physics"
        );
        Ok(Self {
            qpos,
            qvel,
            root_qpos: model.jnt_qposadr()[free] as usize,
            root_qvel: model.jnt_dofadr()[free] as usize,
            root_offset,
            root_rotation,
        })
    }

    fn root(&self, sample: &Sample) -> (DVec3, DQuat, DVec3, DVec3) {
        let p = self.root_qpos;
        let v = self.root_qvel;
        let quaternion = from_wxyz(sample.qpos[p + 3..p + 7].try_into().unwrap()).normalize();
        let offset = quaternion * self.root_offset;
        let angular = quaternion * DVec3::from_slice(&sample.qvel[v + 3..v + 6]);
        (
            DVec3::from_slice(&sample.qpos[p..p + 3]) + offset,
            (quaternion * self.root_rotation).normalize(),
            DVec3::from_slice(&sample.qvel[v..v + 3]) + angular.cross(offset),
            angular,
        )
    }
}

pub struct SuperMimic {
    pub policy: Policy,
    settings: Settings,
    clip: Clip,
    layout: Layout,
    root_from_imu: DQuat,
    alignment: Option<DQuat>,
    history: Vec<f32>,
    last_action: Vec<f32>,
    last_step: Option<u64>,
}

impl SuperMimic {
    pub fn new(mut policy: Policy, clip_name: &str) -> Result<Self> {
        ensure!(
            policy.metadata.task == "supermimic",
            "SuperMimic bundle required"
        );
        let settings: Settings = serde_json::from_value(policy.document.clone())?;
        ensure!(
            settings
                .reference_clips
                .iter()
                .any(|name| name == clip_name),
            "clip is not selected in SuperMimic bundle"
        );
        let clip = Clip::load(
            &policy
                .reference_clips
                .remove(clip_name)
                .context("missing SuperMimic reference clip")?,
        )?;
        let n = policy.metadata.coordinates.len();
        ensure!(
            n > 0
                && settings.joint_names.len() == n
                && clip.joint_names == settings.joint_names
                && settings
                    .joint_names
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == n
                && settings.joint_from_motor.len() == n
                && settings
                    .joint_from_motor
                    .iter()
                    .all(|row| row.len() == n && row.iter().all(|v| v.is_finite()))
                && settings.default_joint_position.len() == n
                && settings
                    .default_joint_position
                    .iter()
                    .all(|v| v.is_finite())
                && policy.metadata.policy_indices == (0..n).collect::<Vec<_>>(),
            "invalid SuperMimic joint mapping"
        );
        ensure!(
            settings.history_frames > 0
                && settings.future_frames > 0
                && settings.history_frames <= 1000
                && settings.future_frames <= 1000
                && settings.goal_joint_velocity_scale.is_finite()
                && settings.goal_joint_velocity_scale > 0.
                && settings.collection_action_clip.is_finite()
                && settings.collection_action_clip > 0.
                && settings.simulation_control == "unprojected_pd"
                && settings.rotation_6d_layout == "first_two_columns_row_major",
            "invalid SuperMimic codec settings"
        );
        let mut sample_layout = vec![
            ("joint_position".to_owned(), n),
            ("joint_velocity".to_owned(), n),
            ("base_angular_velocity".to_owned(), 3),
            ("gravity_direction".to_owned(), 3),
            ("last_action".to_owned(), n),
        ];
        if settings.reference_orientation {
            sample_layout.push(("reference_orientation_6d".to_owned(), 6));
        }
        let size: usize = sample_layout.iter().map(|(_, n)| n).sum();
        ensure!(
            settings.sample_frame == sample_layout
                && policy.metadata.observation_stack_size == 1
                && policy.metadata.observation_frame
                    == vec![
                        (
                            "proprioceptive_history".to_owned(),
                            size * settings.history_frames
                        ),
                        (
                            "future_reference".to_owned(),
                            (15 + 2 * n) * settings.future_frames
                        )
                    ],
            "invalid SuperMimic observation layout"
        );
        let layout = Layout::load(&policy.physics_xml, &settings, &clip)?;
        let root_from_imu = checked(DQuat::from_array(settings.root_from_hardware_imu_xyzw))?;
        Ok(Self {
            policy,
            settings,
            clip,
            layout,
            root_from_imu,
            alignment: None,
            history: Vec::new(),
            last_action: vec![0.; n],
            last_step: None,
        })
    }

    pub fn step(&mut self, feedback: &Feedback) -> Result<Step> {
        let n = self.settings.joint_names.len();
        ensure!(
            feedback.motor_position.len() == n
                && feedback.motor_velocity.len() == n
                && feedback.previous_motor_target.len() == n
                && feedback
                    .motor_position
                    .iter()
                    .chain(&feedback.motor_velocity)
                    .chain(&feedback.previous_motor_target)
                    .chain(&feedback.orientation_xyzw)
                    .chain(&feedback.angular_velocity_local)
                    .all(|v| v.is_finite()),
            "invalid SuperMimic feedback"
        );
        ensure!(
            feedback.reference_time.is_finite()
                && feedback.reference_time >= 0.
                && feedback.reference_time <= self.clip.duration(),
            "invalid SuperMimic reference time"
        );
        let clock = feedback.reference_time / self.policy.metadata.control_timestep;
        ensure!(
            (clock - clock.round_ties_even()).abs() < 1e-4 && clock <= f64::from(1_u32 << 24),
            "SuperMimic reference time must be on the control clock"
        );
        let step = clock.round_ties_even() as u64;
        let frame =
            step as f32 * (self.clip.fps() as f32 * self.policy.metadata.control_timestep as f32);
        let current = self.clip.sample_frame(f64::from(frame));
        let (reference_position, reference_orientation, _, _) = self.layout.root(&current);
        let orientation = (checked(DQuat::from_array(feedback.orientation_xyzw))?
            * self.root_from_imu.conjugate())
        .normalize();
        if feedback.initialize {
            self.history.clear();
            self.last_action.fill(0.);
            self.last_step = None;
            self.alignment =
                Some(heading(orientation) * heading(reference_orientation).conjugate());
        }
        let alignment = self
            .alignment
            .context("initialize SuperMimic before inference")?;
        ensure!(
            self.last_step.is_none_or(|previous| step == previous + 1),
            "SuperMimic reference time must advance one control period"
        );
        let map = |input: &[f64]| -> Vec<f64> {
            self.settings
                .joint_from_motor
                .iter()
                .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
                .collect()
        };
        let position = map(&feedback.motor_position);
        let velocity = map(&feedback.motor_velocity);
        let mut sample = Vec::new();
        sample.extend(
            position
                .iter()
                .zip(&self.settings.default_joint_position)
                .map(|(a, b)| a - b),
        );
        sample.extend(velocity);
        sample.extend(
            (self.root_from_imu * DVec3::from_array(feedback.angular_velocity_local)).to_array(),
        );
        sample.extend((orientation.conjugate() * DVec3::NEG_Z).to_array());
        sample.extend(self.last_action.iter().map(|v| f64::from(*v)));
        if self.settings.reference_orientation {
            sample.extend(rotation_6d(
                orientation.conjugate() * alignment * reference_orientation,
            ));
        }
        let sample: Vec<f32> = sample.into_iter().map(|v| v as f32).collect();
        if self.history.is_empty() {
            self.history = sample.repeat(self.settings.history_frames);
        } else {
            self.history.rotate_left(sample.len());
            let start = self.history.len() - sample.len();
            self.history[start..].copy_from_slice(&sample);
        }
        let mut observation = self.history.clone();
        let inverse_heading = heading(reference_orientation).conjugate();
        for i in 1..=self.settings.future_frames {
            let offset =
                i as f32 * (self.clip.fps() as f32 * self.policy.metadata.control_timestep as f32);
            let future = self.clip.sample_frame(f64::from(frame + offset));
            let (position, quaternion, linear, angular) = self.layout.root(&future);
            observation.extend(
                (inverse_heading * (position - reference_position))
                    .to_array()
                    .map(|v| v as f32),
            );
            observation.extend(rotation_6d(inverse_heading * quaternion).map(|v| v as f32));
            observation.extend(
                (quaternion.conjugate() * linear)
                    .to_array()
                    .map(|v| v as f32),
            );
            observation.extend(
                (quaternion.conjugate() * angular)
                    .to_array()
                    .map(|v| v as f32),
            );
            observation.extend(
                self.layout
                    .qpos
                    .iter()
                    .map(|&index| future.qpos[index] as f32),
            );
            observation.extend(self.layout.qvel.iter().map(|&index| {
                (future.qvel[index] * self.settings.goal_joint_velocity_scale) as f32
            }));
        }
        let output = self.policy.infer(&observation)?;
        let proposal: Vec<f64> = output.motor_target.iter().map(|v| f64::from(*v)).collect();
        let limited = self
            .policy
            .target_limits
            .constrain(&proposal, &feedback.previous_motor_target)?;
        self.last_action = output.action.clone();
        self.last_step = Some(step);
        Ok(Step {
            observation,
            raw_action: output.action,
            action: self.last_action.clone(),
            requested_motor_target: limited.requested_motor_target,
            motor_target: limited.motor_target,
            limited: limited.limited,
        })
    }
}
