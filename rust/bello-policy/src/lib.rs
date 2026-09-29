use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, ensure};
use ort::{
    session::Session,
    tensor::TensorElementType,
    value::{Tensor, ValueType},
};
use serde::{Deserialize, Serialize};

pub mod control;
pub mod following;
pub mod gait;
pub mod hardware;
pub mod locomotion;
pub mod motion;
pub mod operator;
pub mod power;
pub mod sonic;
pub mod target_limits;
pub mod torso;

#[derive(Debug, Deserialize)]
pub struct PolicyMetadata {
    pub git_commit: String,
    pub task: String,
    pub robot: String,
    pub observation_size: usize,
    pub observation_frame: Vec<(String, usize)>,
    pub observation_stack_size: usize,
    pub observation_max_abs: f32,
    pub coordinates: Vec<String>,
    pub policy_indices: Vec<usize>,
    pub default_position: Vec<f32>,
    pub kp: Vec<f32>,
    pub kd: Vec<f32>,
    pub control_timestep: f64,
}

impl PolicyMetadata {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.task.as_str(),
                "locomotion" | "motion_tracking" | "sonic_transfer"
            ),
            "unknown policy task"
        );
        ensure!(
            self.git_commit.len() == 40
                && self
                    .git_commit
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid source commit"
        );
        let count = self.coordinates.len();
        ensure!(
            count > 0
                && self.coordinates.iter().all(|s| !s.is_empty())
                && self.coordinates.iter().collect::<BTreeSet<_>>().len() == count,
            "coordinate names must be nonempty and unique"
        );
        ensure!(
            !self.policy_indices.is_empty()
                && self.policy_indices.iter().all(|&i| i < count)
                && self.policy_indices.iter().collect::<BTreeSet<_>>().len()
                    == self.policy_indices.len(),
            "invalid policy coordinate indices"
        );
        for values in [&self.default_position, &self.kp, &self.kd] {
            ensure!(
                values.len() == count && values.iter().all(|v| v.is_finite()),
                "invalid motor defaults or gains"
            );
        }
        ensure!(
            self.kp.iter().chain(&self.kd).all(|v| *v >= 0.),
            "motor gains must be nonnegative"
        );
        let frame_size = self
            .observation_frame
            .iter()
            .try_fold(0_usize, |size, (_, n)| size.checked_add(*n))
            .context("observation dimensions overflow")?;
        ensure!(
            frame_size > 0
                && self.observation_stack_size > 0
                && frame_size.checked_mul(self.observation_stack_size)
                    == Some(self.observation_size),
            "invalid stacked observation dimensions"
        );
        ensure!(
            self.observation_max_abs.is_finite() && self.observation_max_abs > 0.,
            "invalid observation bound"
        );
        ensure!(
            self.control_timestep.is_finite() && self.control_timestep > 0.,
            "invalid control period"
        );
        Ok(())
    }

    pub fn require_source(&self, source: &Path) -> Result<()> {
        let git = |args: &[&str]| -> Result<String> {
            let result = Command::new("git")
                .arg("-C")
                .arg(source)
                .args(args)
                .output()?;
            ensure!(
                result.status.success(),
                "git check failed: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            Ok(String::from_utf8(result.stdout)?.trim().to_owned())
        };
        ensure!(
            git(&["rev-parse", "HEAD"])? == self.git_commit,
            "policy requires bello_mujoco commit {}",
            self.git_commit
        );
        ensure!(
            git(&[
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignore-submodules=none"
            ])?
            .is_empty(),
            "policy source checkout is dirty"
        );
        let mut submodule_args = vec![
            "submodule",
            "status",
            "--recursive",
            "--",
            "deps/bello-stack",
        ];
        if self.task == "sonic_transfer" {
            submodule_args.push("deps/gear_sonic");
        }
        let submodules = git(&submodule_args)?;
        ensure!(
            !submodules.is_empty()
                && submodules
                    .lines()
                    .all(|line| !line.trim_start().starts_with(['+', '-', 'U'])),
            "runtime submodules must be initialized at their pinned commits"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct PolicyOutput {
    pub action: Vec<f32>,
    pub motor_target: Vec<f32>,
}

pub struct Policy {
    pub target_limits: target_limits::TargetLimits,
    session: Session,
    pub metadata: PolicyMetadata,
    pub reference_motion: Option<Vec<u8>>,
    pub reference_clips: BTreeMap<String, Vec<u8>>,
    pub physics_xml: Vec<u8>,
    pub document: serde_json::Value,
}

fn validate_graph(session: &Session, metadata: &PolicyMetadata) -> Result<()> {
    ensure!(
        session.inputs.len() == 1 && session.outputs.len() == 2,
        "unexpected ONNX inputs/outputs"
    );
    let validate = |name: &str, value: &ValueType, width: usize| -> Result<()> {
        ensure!(
            matches!(value, ValueType::Tensor { ty: TensorElementType::Float32, shape, .. }
            if shape.as_ref() == [1, width as i64]),
            "ONNX {name} shape/type disagrees with policy metadata"
        );
        Ok(())
    };
    let input = &session.inputs[0];
    ensure!(
        input.name == "observation",
        "ONNX observation input is missing"
    );
    validate("observation", &input.input_type, metadata.observation_size)?;
    for (name, width) in [
        ("action", metadata.policy_indices.len()),
        ("motor_target", metadata.coordinates.len()),
    ] {
        let output = session
            .outputs
            .iter()
            .find(|output| output.name == name)
            .with_context(|| format!("ONNX {name} output is missing"))?;
        validate(name, &output.output_type, width)?;
    }
    Ok(())
}

impl Policy {
    pub fn load(path: &Path, source: &Path) -> Result<Self> {
        let mut archive = zip::ZipArchive::new(File::open(path)?)?;
        let mut model = Vec::new();
        archive
            .by_name("policy.onnx")
            .context("deployment bundle is missing policy.onnx")?
            .read_to_end(&mut model)?;
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_memory(&model)?;
        let document = session
            .metadata()?
            .custom("bello")?
            .context("ONNX file has no Bello metadata")?;
        let document: serde_json::Value = serde_json::from_str(&document)?;
        let metadata: PolicyMetadata = serde_json::from_value(document.clone())?;
        metadata.validate()?;
        metadata.require_source(source)?;
        validate_graph(&session, &metadata)?;
        let target_limits =
            target_limits::TargetLimits::new(&document, &metadata.default_position)?;
        let mut physics_xml = Vec::new();
        archive
            .by_name("physics.xml")
            .context("deployment bundle is missing physics.xml")?
            .read_to_end(&mut physics_xml)?;
        let reference_motion = if metadata.task == "motion_tracking" {
            let mut bytes = Vec::new();
            archive
                .by_name("motion.npz")
                .context("motion deployment bundle is missing motion.npz")?
                .read_to_end(&mut bytes)?;
            Some(bytes)
        } else {
            None
        };
        let mut reference_clips = BTreeMap::new();
        if metadata.task == "sonic_transfer" {
            #[derive(Deserialize)]
            struct Manifest {
                clips: Vec<String>,
            }
            let manifest: Manifest = serde_json::from_reader(
                archive
                    .by_name("clips.json")
                    .context("SONIC deployment bundle is missing clips.json")?,
            )?;
            ensure!(
                !manifest.clips.is_empty()
                    && manifest.clips.iter().all(|name| {
                        !name.is_empty()
                            && std::path::Path::new(name)
                                .file_name()
                                .and_then(|s| s.to_str())
                                == Some(name)
                    })
                    && manifest.clips.iter().collect::<BTreeSet<_>>().len() == manifest.clips.len(),
                "invalid SONIC clip manifest"
            );
            for name in manifest.clips {
                let mut bytes = Vec::new();
                archive
                    .by_name(&format!("clips/{name}.npz"))
                    .with_context(|| format!("SONIC bundle is missing clip {name}"))?
                    .read_to_end(&mut bytes)?;
                reference_clips.insert(name, bytes);
            }
        }
        Ok(Self {
            target_limits,
            session,
            metadata,
            reference_motion,
            reference_clips,
            physics_xml,
            document,
        })
    }

    pub fn infer(&mut self, observation: &[f32]) -> Result<PolicyOutput> {
        ensure!(
            observation.len() == self.metadata.observation_size,
            "observation shape mismatch"
        );
        ensure!(
            observation
                .iter()
                .all(|v| v.is_finite() && v.abs() <= self.metadata.observation_max_abs),
            "nonfinite or implausible observation"
        );
        let tensor = Tensor::from_array((
            [1, observation.len()],
            observation.to_vec().into_boxed_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs!["observation" => tensor])?;
        let (shape, action) = outputs
            .get("action")
            .context("ONNX action output is missing")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, self.metadata.policy_indices.len() as i64],
            "action output shape mismatch"
        );
        let action = action.to_vec();
        let (shape, target) = outputs
            .get("motor_target")
            .context("ONNX motor_target output is missing")?
            .try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, self.metadata.coordinates.len() as i64],
            "motor output shape mismatch"
        );
        ensure!(
            action.iter().chain(target).all(|v| v.is_finite()),
            "policy produced nonfinite outputs"
        );
        Ok(PolicyOutput {
            action,
            motor_target: target.to_vec(),
        })
    }
}
