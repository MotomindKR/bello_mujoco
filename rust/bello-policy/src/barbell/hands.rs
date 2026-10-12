//! Independent lease worker: RPC latency never blocks body control.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tonic::transport::Endpoint;

use crate::hands::proto;
use proto::{hand_control_client::HandControlClient, *};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub endpoint: String,
    pub rpc_timeout_seconds: f64,
    pub update_seconds: f64,
    pub stale_seconds: f64,
    pub acquisition_timeout_seconds: f64,
    pub max_raw_rate_per_second: f64,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.rpc_timeout_seconds,
                self.update_seconds,
                self.stale_seconds,
                self.acquisition_timeout_seconds,
                self.max_raw_rate_per_second
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0. && *v < 86400.),
            "invalid barbell hand settings"
        );
        ensure!(
            self.stale_seconds > self.update_seconds + 2. * self.rpc_timeout_seconds
                && self.acquisition_timeout_seconds > self.stale_seconds,
            "inconsistent hand timing"
        );
        Endpoint::from_shared(self.endpoint.clone())?;
        Ok(())
    }
}
#[derive(Clone)]
pub struct State {
    pub updated: Instant,
    pub ready: bool,
    pub closure: f64,
    pub error: String,
}
pub struct Session {
    desired: Option<watch::Sender<f64>>,
    state: watch::Receiver<State>,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    started: Instant,
    config: Config,
}
impl Session {
    pub fn start(config: Config) -> Self {
        let started = Instant::now();
        let (desired, rx) = watch::channel(0.);
        let (tx, state) = watch::channel(State {
            updated: started,
            ready: false,
            closure: 0.,
            error: String::new(),
        });
        let settings = config.clone();
        let task = tokio::spawn(async move {
            let result = worker(settings, rx, tx.clone()).await;
            if let Err(e) = &result {
                tx.send_modify(|s| {
                    s.ready = false;
                    s.error = format!("{e:#}");
                });
            }
            result
        });
        Self {
            desired: Some(desired),
            state,
            task: Some(task),
            started,
            config,
        }
    }
    pub fn command(&self, closure: f64) {
        if let Some(tx) = &self.desired {
            tx.send_if_modified(|v| {
                if *v == closure {
                    false
                } else {
                    *v = closure;
                    true
                }
            });
        }
    }
    pub fn state(&self) -> Result<State> {
        let s = self.state.borrow().clone();
        ensure!(s.error.is_empty(), "barbell hands: {}", s.error);
        if s.ready {
            ensure!(
                s.updated.elapsed().as_secs_f64() < self.config.stale_seconds,
                "barbell hand feedback stale"
            );
        } else {
            ensure!(
                self.started.elapsed().as_secs_f64() < self.config.acquisition_timeout_seconds,
                "barbell hand acquisition timed out"
            );
        }
        Ok(s)
    }
    pub fn begin_close(&mut self) {
        self.desired.take();
    }
    pub fn closing(&self) -> bool {
        self.desired.is_none()
    }
    pub fn finished(&self) -> bool {
        self.task.as_ref().is_none_or(|task| task.is_finished())
    }
    pub async fn close(&mut self) -> Result<()> {
        self.desired.take();
        if let Some(mut task) = self.task.take() {
            match tokio::time::timeout(
                Duration::from_secs_f64(
                    8. * self.config.rpc_timeout_seconds + self.config.update_seconds,
                ),
                &mut task,
            )
            .await
            {
                Ok(result) => result??,
                Err(error) => {
                    task.abort();
                    return Err(error)
                        .context("hand release timed out; leases will expire in HOLD");
                }
            }
        }
        Ok(())
    }
}

fn validate(status: &Status, closure: f64) -> Result<()> {
    for side in [Side::Left, Side::Right] {
        let hand = status
            .hands
            .iter()
            .find(|h| h.side == side as i32)
            .context("missing barbell hand")?;
        ensure!(
            hand.owned
                && hand.feedback_fresh
                && hand.transport_fault.is_empty()
                && hand.suspension_reason.is_empty()
                && hand.source == Source::Application as i32
                && hand.target.len() == 6
                && hand
                    .target
                    .iter()
                    .all(|v| v.is_finite() && (*v - closure).abs() < 1e-6)
                && hand.measured_positions.len() == 6
                && hand.commanded_positions.len() == 6,
            "hand {:?} lost ownership, intent or fresh feedback: {} {}",
            side,
            hand.suspension_reason,
            hand.transport_fault
        );
    }
    Ok(())
}
async fn worker(
    config: Config,
    mut desired: watch::Receiver<f64>,
    state: watch::Sender<State>,
) -> Result<()> {
    let timeout = Duration::from_secs_f64(config.rpc_timeout_seconds);
    let channel = Endpoint::from_shared(config.endpoint.clone())?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await?;
    let mut client = HandControlClient::new(channel);
    let mut leases = Vec::new();
    let operation: Result<()> = async {
        let status = client.get_status(Empty {}).await?.into_inner();
        ensure!(
            status.lease_timeout_seconds.is_finite()
                && status.lease_timeout_seconds
                    > config.stale_seconds
                        + config.update_seconds
                        + 2. * config.rpc_timeout_seconds,
            "hand lease timeout is too short for configured renewal deadlines"
        );
        for side in [Side::Left, Side::Right] {
            let lease = client
                .acquire(Hand { side: side as i32 })
                .await?
                .into_inner();
            ensure!(
                lease.side == side as i32 && !lease.token.is_empty(),
                "invalid hand lease"
            );
            leases.push(lease);
        }
        let mut initialized = false;
        loop {
            if desired.has_changed().is_err() {
                break;
            }
            let closure = *desired.borrow_and_update();
            let mut last = None;
            for lease in &leases {
                last = Some(
                    client
                        .set_command(Command {
                            side: lease.side,
                            token: lease.token.clone(),
                            source: Source::Application as i32,
                            target: Some(command::Target::Positions(Positions {
                                values: vec![closure; 6],
                            })),
                            max_raw_rate_per_second: Some(config.max_raw_rate_per_second),
                        })
                        .await?
                        .into_inner(),
                );
            }
            let status = last.as_ref().unwrap();
            if !initialized
                && status
                    .hands
                    .iter()
                    .any(|h| !h.suspension_reason.is_empty() && h.transport_fault.is_empty())
            {
                tokio::time::sleep(Duration::from_secs_f64(config.update_seconds)).await;
                continue;
            }
            validate(status, closure)?;
            initialized = true;
            state.send_replace(State {
                updated: Instant::now(),
                ready: true,
                closure,
                error: String::new(),
            });
            tokio::select! {
                _=tokio::time::sleep(Duration::from_secs_f64(config.update_seconds)) => {},
                changed=desired.changed() => {if changed.is_err() {break;}},
            }
        }
        Ok(())
    }
    .await;
    // Best effort both sides, including a partial acquisition. Never command open.
    let mut release_errors = Vec::new();
    for lease in leases {
        if let Err(e) = client.release(lease).await {
            release_errors.push(e.to_string());
        }
    }
    operation?;
    ensure!(
        release_errors.is_empty(),
        "hand release failed: {release_errors:?}"
    );
    Ok(())
}
