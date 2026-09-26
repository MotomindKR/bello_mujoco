//! One latest-only body reference per policy tick, including torso intent.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};
use tokio::sync::watch;

pub mod proto {
    tonic::include_proto!("bello.reference.v1");
}
use proto::{
    GetReferenceRequest, GetReferenceResponse, reference_service_client::ReferenceServiceClient,
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub enabled: bool,
    pub endpoint: String,
    pub rpc_timeout_seconds: f64,
    pub poll_period_seconds: f64,
    pub max_reference_age_ms: f64,
    pub max_reference_joint_speed_rps: f64,
    pub min_live_joint_limit_margin_radians: f64,
    pub max_target_speed_rps: f64,
    pub connection_transition_duration_seconds: f64,
}

#[derive(Deserialize)]
struct Mapping {
    joint_names: Vec<String>,
    joint_limits: Vec<[f64; 2]>,
    motor_limits: Vec<[f64; 2]>,
    motor_from_joint: Vec<Vec<f64>>,
    joint_from_motor: Vec<Vec<f64>>,
    default_joint_position: Vec<f64>,
}

#[derive(Clone)]
struct Sample {
    message: GetReferenceResponse,
    requested: Instant,
}

pub struct Following {
    config: Config,
    mapping: Mapping,
    robot: String,
    indices: Vec<usize>,
    latest: watch::Receiver<Result<Option<Sample>, String>>,
    task: tokio::task::JoinHandle<()>,
    transition: Transition,
    last_sequence: Option<u64>,
}

#[derive(Default)]
struct Transition {
    target: Vec<f64>,
    live: bool,
    start: Option<Vec<f64>>,
    elapsed: f64,
}
impl Transition {
    fn advance(
        &mut self,
        measured: &[f64],
        desired: &[f64],
        live: bool,
        dt: f64,
        duration: f64,
        speed: f64,
    ) -> &[f64] {
        if self.target.is_empty() {
            self.target = measured.to_vec();
        }
        if live != self.live {
            self.live = live;
            self.start = Some(self.target.clone());
            self.elapsed = 0.;
        }
        self.elapsed = (self.elapsed + dt).min(duration);
        for (i, value) in self.target.iter_mut().enumerate() {
            let requested = self.start.as_ref().map_or(desired[i], |start| {
                start[i] + self.elapsed / duration * (desired[i] - start[i])
            });
            *value += (requested - *value).clamp(-speed * dt, speed * dt);
        }
        if self.elapsed >= duration {
            self.start = None;
        }
        &self.target
    }
}

fn multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

impl Following {
    pub fn new(config: Config, policy: &crate::Policy) -> Result<Self> {
        for value in [
            config.rpc_timeout_seconds,
            config.poll_period_seconds,
            config.max_reference_age_ms,
            config.max_reference_joint_speed_rps,
            config.max_target_speed_rps,
            config.connection_transition_duration_seconds,
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 60000.,
                "invalid following timing/limit"
            );
        }
        ensure!(
            config.min_live_joint_limit_margin_radians.is_finite()
                && config.min_live_joint_limit_margin_radians >= 0.,
            "invalid following joint margin"
        );
        let mapping: Mapping = serde_json::from_value(policy.document.clone())?;
        let count = policy.metadata.coordinates.len();
        ensure!(
            mapping.joint_names.len() == count
                && mapping.joint_names.iter().all(|s| !s.is_empty())
                && mapping.joint_names.iter().collect::<BTreeSet<_>>().len() == count,
            "invalid following joint names"
        );
        for matrix in [&mapping.motor_from_joint, &mapping.joint_from_motor] {
            ensure!(
                matrix.len() == count
                    && matrix
                        .iter()
                        .all(|r| r.len() == count && r.iter().all(|v| v.is_finite())),
                "invalid following coordinate mapping"
            );
        }
        for i in 0..count {
            for j in 0..count {
                let value: f64 = (0..count)
                    .map(|k| mapping.motor_from_joint[i][k] * mapping.joint_from_motor[k][j])
                    .sum();
                ensure!(
                    (value - if i == j { 1. } else { 0. }).abs() < 1e-8,
                    "following coordinate mappings are not inverses"
                );
            }
        }
        for ranges in [&mapping.joint_limits, &mapping.motor_limits] {
            ensure!(
                ranges.len() == count
                    && ranges
                        .iter()
                        .all(|r| r[0].is_finite() && r[1].is_finite() && r[0] < r[1]),
                "invalid following coordinate limits"
            );
        }
        ensure!(
            mapping.default_joint_position.len() == count
                && mapping.default_joint_position.iter().all(|v| v.is_finite()),
            "invalid following nominal pose"
        );
        let indices: Vec<_> = (0..count)
            .filter(|i| !policy.metadata.policy_indices.contains(i))
            .collect();
        ensure!(
            !indices.is_empty(),
            "policy has no held coordinates for following"
        );
        for &motor in &policy.metadata.policy_indices {
            ensure!(
                indices
                    .iter()
                    .all(|&joint| mapping.motor_from_joint[motor][joint].abs() < 1e-12),
                "following joints overlap policy motor ownership"
            );
        }
        let robot = policy.metadata.robot.clone();
        let endpoint = tonic::transport::Endpoint::from_shared(config.endpoint.clone())?
            .connect_timeout(Duration::from_secs_f64(config.rpc_timeout_seconds));
        let timeout = Duration::from_secs_f64(config.rpc_timeout_seconds);
        let period = Duration::from_secs_f64(config.poll_period_seconds);
        let request = GetReferenceRequest {
            robot: robot.clone(),
        };
        let (tx, latest) = watch::channel(Ok(None));
        let task =
            tokio::spawn(async move {
                let mut client = ReferenceServiceClient::new(endpoint.connect_lazy())
                    .max_decoding_message_size(64 * 1024);
                loop {
                    let requested = Instant::now();
                    let sample =
                        match tokio::time::timeout(timeout, client.get_reference(request.clone()))
                            .await
                        {
                            Ok(Ok(response)) => Ok(Some(Sample {
                                message: response.into_inner(),
                                requested,
                            })),
                            Err(_) => Ok(None),
                            Ok(Err(status))
                                if matches!(
                                    status.code(),
                                    tonic::Code::Unavailable | tonic::Code::DeadlineExceeded
                                ) =>
                            {
                                Ok(None)
                            }
                            Ok(Err(status)) => Err(format!("reference producer failed: {status}")),
                        };
                    let terminal = sample.is_err();
                    let _ = tx.send_replace(sample);
                    if terminal {
                        return;
                    }
                    tokio::time::sleep(period.saturating_sub(requested.elapsed())).await;
                }
            });
        Ok(Self {
            config,
            mapping,
            robot,
            indices,
            latest,
            task,
            transition: Transition::default(),
            last_sequence: None,
        })
    }

    pub fn update(&mut self, measured: &[f64], dt: f64) -> Result<(bool, [f32; 3])> {
        let m = &self.mapping;
        ensure!(
            measured.len() == m.joint_names.len(),
            "following feedback shape mismatch"
        );
        let sample = self.latest.borrow().clone().map_err(anyhow::Error::msg)?;
        let mut desired: Vec<_> = self
            .indices
            .iter()
            .map(|&i| m.default_joint_position[i])
            .collect();
        let mut live = false;
        let mut torso = [0.; 3];
        if let Some(sample) = sample {
            let msg = sample.message;
            ensure!(msg.robot == self.robot, "reference producer robot mismatch");
            ensure!(
                msg.source_age_ms.is_finite() && msg.source_age_ms >= 0.,
                "invalid reference age"
            );
            if msg.available {
                ensure!(
                    self.last_sequence.is_none_or(|s| msg.sequence >= s),
                    "reference sequence regressed"
                );
                self.last_sequence = Some(msg.sequence);
                let names: Vec<_> = self.indices.iter().map(|&i| &m.joint_names[i]).collect();
                ensure!(
                    msg.joint_names.len() == names.len()
                        && msg.joint_names.iter().collect::<BTreeSet<_>>()
                            == names.into_iter().collect()
                        && msg.position.len() == msg.joint_names.len()
                        && msg.velocity.len() == msg.joint_names.len()
                        && msg
                            .position
                            .iter()
                            .chain(&msg.velocity)
                            .all(|v| v.is_finite()),
                    "invalid reference joint names/shape/values"
                );
                ensure!(
                    msg.torso_command.len() == 3
                        && msg.torso_command.iter().all(|x| (*x as f32).is_finite()),
                    "invalid reference torso shape/values"
                );
                live = msg.source_age_ms + sample.requested.elapsed().as_secs_f64() * 1000.
                    <= self.config.max_reference_age_ms;
                for (slot, &joint) in self.indices.iter().enumerate() {
                    let index = msg
                        .joint_names
                        .iter()
                        .position(|n| n == &m.joint_names[joint])
                        .context("missing reference joint")?;
                    let q = msg.position[index];
                    let [low, high] = m.joint_limits[joint];
                    live &= msg.velocity[index].abs() <= self.config.max_reference_joint_speed_rps
                        && (q - low).min(high - q)
                            > self.config.min_live_joint_limit_margin_radians;
                    desired[slot] = q;
                }
                if live {
                    torso = std::array::from_fn(|i| msg.torso_command[i] as f32);
                } else {
                    for (slot, &joint) in self.indices.iter().enumerate() {
                        desired[slot] = m.default_joint_position[joint];
                    }
                }
            } else {
                self.last_sequence = None;
            }
        } else {
            self.last_sequence = None;
        }
        let measured = multiply(&m.joint_from_motor, measured);
        let measured: Vec<_> = self.indices.iter().map(|&i| measured[i]).collect();
        self.transition.advance(
            &measured,
            &desired,
            live,
            dt,
            self.config.connection_transition_duration_seconds,
            self.config.max_target_speed_rps,
        );
        Ok((live, torso))
    }

    pub fn apply(&self, target: &mut [f32]) -> Result<()> {
        let m = &self.mapping;
        ensure!(
            target.len() == m.joint_names.len()
                && self.transition.target.len() == self.indices.len(),
            "following target shape mismatch or reference not sampled"
        );
        let values: Vec<_> = target.iter().map(|v| *v as f64).collect();
        let mut logical = multiply(&m.joint_from_motor, &values);
        for (&joint, &value) in self.indices.iter().zip(&self.transition.target) {
            logical[joint] = value;
        }
        for ((value, result), [low, high]) in target
            .iter_mut()
            .zip(multiply(&m.motor_from_joint, &logical))
            .zip(&m.motor_limits)
        {
            *value = result.clamp(*low, *high) as f32;
        }
        Ok(())
    }
}
impl Drop for Following {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outage_returns_to_nominal_without_target_jumps() {
        let mut state = Transition::default();
        let a = state.advance(&[0.], &[2.], true, 0.1, 1., 0.5)[0];
        assert!((a - 0.05).abs() < 1e-12);
        let b = state.advance(&[0.], &[0.], false, 0.1, 1., 0.5)[0];
        assert!((b - 0.045).abs() < 1e-12);
        for _ in 0..30 {
            state.advance(&[0.], &[0.], false, 0.1, 1., 0.5);
        }
        assert!(state.target[0].abs() < 1e-12);
    }
}
