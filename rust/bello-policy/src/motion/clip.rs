//! Read the training clip directly; no relocated assets or compatibility conversion.

use std::io::Cursor;

use anyhow::{Result, ensure};
use glam::{DQuat, DVec3};

use super::rotation::{checked_quaternion, from_wxyz, to_wxyz};

pub struct Clip {
    pub joint_names: Vec<String>,
    pub body_names: Vec<String>,
    fps: f64,
    frames: usize,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    body_pos: Vec<f64>,
    body_quat: Vec<DQuat>,
}

pub struct Sample {
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub body_pos: Vec<DVec3>,
    pub body_quat: Vec<DQuat>,
}

type Archive<'a> = zip::ZipArchive<Cursor<&'a [u8]>>;

fn array<T: npyz::Deserialize>(
    archive: &mut Archive<'_>,
    name: &str,
) -> Result<(Vec<u64>, Vec<T>)> {
    let file = npyz::NpyFile::new(archive.by_name(&format!("{name}.npy"))?)?;
    ensure!(file.order() == npyz::Order::C, "{name} must be row-major");
    Ok((file.shape().to_vec(), file.into_vec::<T>()?))
}

fn floats(archive: &mut Archive<'_>, name: &str, shape: &[u64]) -> Result<Vec<f64>> {
    let (actual, values) = array::<f32>(archive, name)?;
    ensure!(
        actual == shape,
        "invalid {name} shape: {actual:?}, expected {shape:?}"
    );
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite clip {name}"
    );
    Ok(values.into_iter().map(f64::from).collect())
}

impl Clip {
    pub fn load(bytes: &[u8]) -> Result<Self> {
        let mut archive = Archive::new(Cursor::new(bytes))?;
        let (shape, fps) = array::<f64>(&mut archive, "fps")?;
        ensure!(
            shape.is_empty() && fps.len() == 1 && fps[0].is_finite() && fps[0] > 0.,
            "invalid clip fps"
        );
        let mut names = |name| -> Result<Vec<String>> {
            let (shape, values) = array::<String>(&mut archive, name)?;
            ensure!(
                shape == [values.len() as u64]
                    && !values.is_empty()
                    && values.iter().all(|s| !s.is_empty())
                    && values
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        == values.len(),
                "invalid clip {name}"
            );
            Ok(values)
        };
        let joint_names = names("joint_names")?;
        let body_names = names("body_names")?;
        let shape = {
            let file = npyz::NpyFile::new(archive.by_name("qpos.npy")?)?;
            file.shape().to_vec()
        };
        ensure!(
            shape.len() == 2 && shape[0] >= 2,
            "clip requires at least two frames"
        );
        let frames = shape[0];
        let duration = (frames - 1) as f64 / fps[0];
        ensure!(
            duration.is_finite() && duration > 0.,
            "invalid clip duration"
        );
        let joints = joint_names.len() as u64;
        let bodies = body_names.len() as u64;
        let mut qpos = floats(&mut archive, "qpos", &[frames, 7 + joints])?;
        let qvel = floats(&mut archive, "qvel", &[frames, 6 + joints])?;
        let body_pos = floats(&mut archive, "body_pos", &[frames, bodies, 3])?;
        let body_quat = floats(&mut archive, "body_quat_wxyz", &[frames, bodies, 4])?
            .chunks_exact(4)
            .map(|q| checked_quaternion(from_wxyz(q.try_into().unwrap())))
            .collect::<Result<Vec<_>>>()?;
        for row in qpos.chunks_exact_mut(7 + joints as usize) {
            let quat = checked_quaternion(from_wxyz(row[3..7].try_into()?))?;
            row[3..7].copy_from_slice(&to_wxyz(quat));
        }
        Ok(Self {
            joint_names,
            body_names,
            fps: fps[0],
            frames: frames as usize,
            qpos,
            qvel,
            body_pos,
            body_quat,
        })
    }

    pub fn duration(&self) -> f64 {
        (self.frames - 1) as f64 / self.fps
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn sample(&self, time: f64) -> Sample {
        self.sample_frame(time * self.fps)
    }

    pub fn sample_frame(&self, frame: f64) -> Sample {
        let frame = frame.clamp(0., (self.frames - 1) as f64);
        let lower = frame.floor() as usize;
        let upper = (lower + 1).min(self.frames - 1);
        let alpha = frame - lower as f64;
        let lerp = |values: &[f64], width| -> Vec<f64> {
            (0..width)
                .map(|i| {
                    (1. - alpha) * values[lower * width + i] + alpha * values[upper * width + i]
                })
                .collect()
        };
        let nq = 7 + self.joint_names.len();
        let mut qpos = lerp(&self.qpos, nq);
        let a = from_wxyz(
            self.qpos[lower * nq + 3..lower * nq + 7]
                .try_into()
                .unwrap(),
        );
        let b = from_wxyz(
            self.qpos[upper * nq + 3..upper * nq + 7]
                .try_into()
                .unwrap(),
        );
        qpos[3..7].copy_from_slice(&to_wxyz(a.slerp(b, alpha).normalize()));
        let bodies = self.body_names.len();
        let body_quat = (0..bodies)
            .map(|i| {
                self.body_quat[lower * bodies + i]
                    .slerp(self.body_quat[upper * bodies + i], alpha)
                    .normalize()
            })
            .collect();
        Sample {
            qpos,
            qvel: lerp(&self.qvel, nq - 1),
            body_pos: lerp(&self.body_pos, bodies * 3)
                .chunks_exact(3)
                .map(DVec3::from_slice)
                .collect(),
            body_quat,
        }
    }
}
