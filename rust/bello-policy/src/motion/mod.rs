//! BeyondMimic observations from motor telemetry and measured floating-base pose.
//! The bundled physics model is used only for kinematics, never as an emulator.

mod clip;
pub mod live;
mod pose;
mod rotation;

use anyhow::{Context, Result, ensure};
use glam::{DQuat, DVec2, DVec3, EulerRot};
use mujoco_rs::prelude::{MjData, MjModel, MjtJoint, MjtObj};
use serde::{Deserialize, Serialize};

use crate::Policy;
use clip::{Clip, Sample};
use rotation::*;

#[derive(Deserialize)]
struct Settings {
    joint_names: Vec<String>,
    joint_from_motor: Vec<Vec<f64>>,
    default_joint_position: Vec<f64>,
    end_effector_names: Vec<String>,
    action_clip: f32,
    environment: Environment,
}

#[derive(Deserialize)]
struct Environment {
    anchor_body_name: String,
    done: Thresholds,
}

#[derive(Deserialize)]
struct Thresholds {
    root_height_error: f64,
    root_orientation_error: f64,
    end_effector_position_error: f64,
}

#[derive(Deserialize)]
pub struct Feedback {
    pub motor_position: Vec<f64>,
    pub motor_velocity: Vec<f64>,
    pub position: [f64; 3],
    pub orientation_xyzw: [f64; 4],
    pub linear_velocity_world: [f64; 3],
    pub angular_velocity_local: [f64; 3],
    pub reference_time: f64,
    /// Begin a clip here and align its starting root XY/yaw to this pose.
    pub initialize: bool,
}

#[derive(Serialize)]
pub struct Step {
    pub observation: Vec<f32>,
    pub raw_action: Vec<f32>,
    pub action: Vec<f32>,
    pub motor_target: Vec<f32>,
}

pub struct Motion {
    pub policy: Policy,
    settings: Settings,
    clip: Clip,
    data: MjData<Box<MjModel>>,
    anchor: usize,
    bodies: Vec<usize>,
    end_effectors: Vec<usize>,
    alignment: Option<(DVec2, DQuat)>,
    history: Vec<f32>,
    last_action: Vec<f32>,
}

impl Motion {
    pub fn new(policy: Policy) -> Result<Self> {
        ensure!(
            policy.metadata.task == "motion_tracking",
            "motion bundle required"
        );
        let settings: Settings = serde_json::from_value(policy.document.clone())?;
        let n = policy.metadata.coordinates.len();
        ensure!(
            settings.joint_names.len() == n
                && settings.joint_from_motor.len() == n
                && settings
                    .joint_from_motor
                    .iter()
                    .all(|row| row.len() == n && row.iter().all(|v| v.is_finite()))
                && settings.default_joint_position.len() == n
                && settings
                    .default_joint_position
                    .iter()
                    .all(|v| v.is_finite()),
            "invalid motion motor mapping"
        );
        ensure!(
            settings.action_clip.is_finite() && settings.action_clip > 0.,
            "invalid action bound"
        );
        let done = &settings.environment.done;
        ensure!(
            [
                done.root_height_error,
                done.root_orientation_error,
                done.end_effector_position_error
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.),
            "invalid termination thresholds"
        );
        let count = policy.metadata.policy_indices.len();
        let layout = [
            ("reference_joint_position", count),
            ("reference_joint_velocity", count),
            ("anchor_position_error", 3),
            ("anchor_orientation_error_6d", 6),
            ("base_linear_velocity", 3),
            ("base_angular_velocity", 3),
            ("joint_position", count),
            ("joint_velocity", count),
            ("last_action", count),
        ];
        ensure!(
            policy
                .metadata
                .observation_frame
                .iter()
                .map(|(s, n)| (s.as_str(), *n))
                .eq(layout),
            "unsupported motion observation layout"
        );
        let clip = Clip::load(
            policy
                .reference_motion
                .as_deref()
                .context("missing motion clip")?,
        )?;
        ensure!(
            clip.joint_names == settings.joint_names,
            "clip and policy joint orders differ"
        );
        ensure!(
            unsafe { mujoco_rs::mujoco_c::mj_version() }
                == mujoco_rs::mujoco_c::mjVERSION_HEADER as i32,
            "loaded MuJoCo must match the pinned 3.12.0 bindings"
        );
        let model = MjModel::from_xml_string(std::str::from_utf8(&policy.physics_xml)?)?;
        ensure!(
            model.nq() as usize == n + 7
                && model.nv() as usize == n + 6
                && model.jnt_type()[0] == MjtJoint::mjJNT_FREE,
            "motion model must have one floating base and scalar joints"
        );
        for (i, name) in settings.joint_names.iter().enumerate() {
            let id = model
                .name_to_id(MjtObj::mjOBJ_JOINT, name)
                .with_context(|| format!("missing motion joint {name}"))?;
            ensure!(
                matches!(
                    model.jnt_type()[id],
                    MjtJoint::mjJNT_HINGE | MjtJoint::mjJNT_SLIDE
                ) && model.jnt_qposadr()[id] as usize == 7 + i
                    && model.jnt_dofadr()[id] as usize == 6 + i,
                "clip/model joint layout mismatch: {name}"
            );
        }
        let bodies = clip
            .body_names
            .iter()
            .map(|name| {
                model
                    .name_to_id(MjtObj::mjOBJ_BODY, name)
                    .with_context(|| format!("missing tracked body {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let index = |name: &str| {
            clip.body_names
                .iter()
                .position(|s| s == name)
                .with_context(|| format!("clip does not track {name}"))
        };
        let anchor = index(&settings.environment.anchor_body_name)?;
        let end_effectors = settings
            .end_effector_names
            .iter()
            .map(|name| index(name))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !end_effectors.is_empty(),
            "motion needs tracked end effectors"
        );
        let data = MjData::try_new(Box::new(model))?;
        Ok(Self {
            policy,
            settings,
            clip,
            data,
            anchor,
            bodies,
            end_effectors,
            alignment: None,
            history: Vec::new(),
            last_action: vec![0.; count],
        })
    }

    pub fn duration(&self) -> f64 {
        self.clip.duration()
    }

    pub fn root_frame(&self) -> Result<&str> {
        let model = self.data.model();
        model
            .id_to_name(MjtObj::mjOBJ_BODY, model.jnt_bodyid()[0] as usize)
            .context("motion root body must be named")
    }

    pub fn root_from_fixed_frame(&mut self, name: &str) -> Result<DQuat> {
        let model = self.data.model();
        let root = model.jnt_bodyid()[0] as usize;
        let frame = model
            .name_to_id(MjtObj::mjOBJ_BODY, name)
            .context("IMU frame is missing from motion model")?;
        let mut body = frame;
        while body != root {
            ensure!(
                body != 0 && model.body_jntnum()[body] == 0,
                "motion IMU frame must be fixed to the root body"
            );
            body = model.body_parentid()[body] as usize;
        }
        self.data.kinematics();
        Ok(from_wxyz(self.data.xquat()[root]).conjugate() * from_wxyz(self.data.xquat()[frame]))
    }

    pub fn reset(&mut self) {
        self.alignment = None;
        self.history.clear();
        self.last_action.fill(0.);
    }

    fn reference(&self, time: f64) -> Sample {
        let mut sample = self.clip.sample(time);
        let (xy, rotation) = self
            .alignment
            .expect("alignment checked before reference sampling");
        for (position, quat) in sample.body_pos.iter_mut().zip(&mut sample.body_quat) {
            *position = rotation * *position;
            position[0] += xy[0];
            position[1] += xy[1];
            *quat = rotation * *quat;
        }
        sample
    }

    pub fn step(&mut self, feedback: &Feedback) -> Result<Step> {
        let n = self.settings.joint_names.len();
        ensure!(
            feedback.motor_position.len() == n && feedback.motor_velocity.len() == n,
            "motion motor feedback shape mismatch"
        );
        ensure!(
            feedback
                .motor_position
                .iter()
                .chain(&feedback.motor_velocity)
                .chain(&feedback.position)
                .chain(&feedback.linear_velocity_world)
                .chain(&feedback.angular_velocity_local)
                .all(|v| v.is_finite()),
            "nonfinite motion feedback"
        );
        ensure!(
            feedback.reference_time.is_finite() && feedback.reference_time >= 0.,
            "invalid reference time"
        );
        ensure!(feedback.reference_time < self.duration(), "motion complete");
        let quat = checked_quaternion(DQuat::from_array(feedback.orientation_xyzw))?;
        if feedback.initialize {
            self.reset();
            let source = self.clip.sample(feedback.reference_time);
            let source_quat = from_wxyz(source.qpos[3..7].try_into()?);
            let rotation = DQuat::from_rotation_z(
                quat.to_euler(EulerRot::ZYX).0 - source_quat.to_euler(EulerRot::ZYX).0,
            );
            let position = rotation * DVec3::from_slice(&source.qpos[..3]);
            self.alignment = Some((
                (DVec3::from_array(feedback.position) - position).truncate(),
                rotation,
            ));
        }
        ensure!(
            self.alignment.is_some(),
            "initialize motion alignment before inference"
        );
        let map = |values: &[f64]| -> Vec<f64> {
            self.settings
                .joint_from_motor
                .iter()
                .map(|row| row.iter().zip(values).map(|(a, b)| a * b).sum())
                .collect()
        };
        let position = map(&feedback.motor_position);
        let velocity = map(&feedback.motor_velocity);
        ensure!(
            position.iter().chain(&velocity).all(|v| v.is_finite()),
            "nonfinite mapped motion feedback"
        );
        self.data.qpos_mut()[..3].copy_from_slice(&feedback.position);
        self.data.qpos_mut()[3..7].copy_from_slice(&to_wxyz(quat));
        self.data.qpos_mut()[7..].copy_from_slice(&position);
        self.data.kinematics();
        let anchor_pos = DVec3::from_array(self.data.xpos()[self.bodies[self.anchor]]);
        let anchor_quat = from_wxyz(self.data.xquat()[self.bodies[self.anchor]]);
        ensure!(
            self.data
                .xpos()
                .iter()
                .flatten()
                .chain(self.data.xquat().iter().flatten())
                .all(|v| v.is_finite()),
            "nonfinite motion kinematics"
        );
        let reference = self.reference(feedback.reference_time);
        let thresholds = &self.settings.environment.done;
        ensure!(
            (anchor_pos[2] - reference.body_pos[self.anchor][2]).abs()
                <= thresholds.root_height_error,
            "motion termination: anchor height"
        );
        let gravity = |q: DQuat| (q.conjugate() * DVec3::NEG_Z).z;
        ensure!(
            (gravity(anchor_quat) - gravity(reference.body_quat[self.anchor])).abs()
                <= thresholds.root_orientation_error,
            "motion termination: anchor orientation"
        );
        // Relative-body targets use yaw-only alignment; their world Z is unchanged.
        ensure!(
            self.end_effectors
                .iter()
                .all(
                    |&i| (self.data.xpos()[self.bodies[i]][2] - reference.body_pos[i][2]).abs()
                        <= thresholds.end_effector_position_error
                ),
            "motion termination: end effector height"
        );
        let reference =
            self.reference(feedback.reference_time + self.policy.metadata.control_timestep);
        let indices = &self.policy.metadata.policy_indices;
        let default = &self.settings.default_joint_position;
        let mut frame = Vec::<f64>::new();
        frame.extend(indices.iter().map(|&i| reference.qpos[7 + i] - default[i]));
        frame.extend(indices.iter().map(|&i| reference.qvel[6 + i]));
        frame.extend(
            (anchor_quat.conjugate() * (reference.body_pos[self.anchor] - anchor_pos)).to_array(),
        );
        frame.extend(rotation_6d(
            anchor_quat.conjugate() * reference.body_quat[self.anchor],
        ));
        frame.extend(
            (quat.conjugate() * DVec3::from_array(feedback.linear_velocity_world)).to_array(),
        );
        frame.extend(feedback.angular_velocity_local);
        frame.extend(indices.iter().map(|&i| position[i] - default[i]));
        frame.extend(indices.iter().map(|&i| velocity[i]));
        frame.extend(self.last_action.iter().map(|&v| f64::from(v)));
        let frame: Vec<f32> = frame.into_iter().map(|v| v as f32).collect();
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
            .map(|a| a.clamp(-self.settings.action_clip, self.settings.action_clip))
            .collect();
        Ok(Step {
            observation: self.history.clone(),
            raw_action: output.action,
            action: self.last_action.clone(),
            motor_target: output.motor_target,
        })
    }
}
