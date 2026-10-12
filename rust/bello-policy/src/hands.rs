//! Runner-owned tracking leases, independent of body command transmission.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tonic::transport::Endpoint;

pub mod proto {
    tonic::include_proto!("bello.hand.v1");
}
use proto::{hand_control_client::HandControlClient, *};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub endpoint: String,
    pub rpc_timeout_seconds: f64,
    pub renew_period_seconds: f64,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        Endpoint::from_shared(self.endpoint.clone())?;
        ensure!(
            [self.rpc_timeout_seconds, self.renew_period_seconds]
                .iter()
                .all(|v| v.is_finite() && *v > 0. && *v < 60.),
            "invalid hand tracking timing"
        );
        Ok(())
    }
}

pub struct Tracking {
    desired: Option<watch::Sender<Option<Instant>>>,
    task: tokio::task::JoinHandle<()>,
    timeout: Duration,
}
impl Tracking {
    pub fn start(config: Config, validity: Duration) -> Self {
        let (desired, rx) = watch::channel(None);
        let timeout = Duration::from_secs_f64(4. * config.rpc_timeout_seconds);
        let task = tokio::spawn(worker(config, rx, validity));
        Self {
            desired: Some(desired),
            task,
            timeout,
        }
    }
    pub fn set_enabled(&self, enabled: bool) {
        if let Some(tx) = &self.desired {
            if enabled {
                tx.send_replace(Some(Instant::now()));
            } else {
                tx.send_if_modified(|value| value.take().is_some());
            }
        }
    }
    pub async fn close(&mut self) -> Result<()> {
        self.desired.take();
        match tokio::time::timeout(self.timeout, &mut self.task).await {
            Ok(result) => result.context("hand tracking worker failed")?,
            Err(_) => {
                self.task.abort();
                anyhow::bail!("hand tracking release timed out; leases will expire in HOLD");
            }
        }
        Ok(())
    }
}

fn fresh(desired: &watch::Receiver<Option<Instant>>, validity: Duration) -> bool {
    desired.has_changed().is_ok()
        && desired
            .borrow()
            .is_some_and(|stamp| stamp.elapsed() < validity)
}

// Mode changes wake the worker immediately; repeated intent updates do not
// increase the hand RPC rate. A stalled body loop cannot maintain hand leases.
async fn wait(
    desired: &mut watch::Receiver<Option<Instant>>,
    validity: Duration,
    until: tokio::time::Instant,
) -> bool {
    loop {
        let stamp = *desired.borrow_and_update();
        let Some(stamp) = stamp.filter(|stamp| stamp.elapsed() < validity) else {
            return false;
        };
        tokio::select! {
            biased;
            changed = desired.changed() => if changed.is_err() { return false; },
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(stamp + validity)) => return false,
            _ = tokio::time::sleep_until(until) => return true,
        }
    }
}

async fn worker(config: Config, mut desired: watch::Receiver<Option<Instant>>, validity: Duration) {
    loop {
        while !fresh(&desired, validity) {
            if desired.changed().await.is_err() {
                return;
            }
        }
        if let Err(error) = session(&config, &mut desired, validity).await {
            eprintln!("Hand tracking suspended: {error:#}");
            // Retry under current operator intent, including after a server restart.
            wait(
                &mut desired,
                validity,
                tokio::time::Instant::now() + Duration::from_secs(1),
            )
            .await;
        }
    }
}

async fn session(
    config: &Config,
    desired: &mut watch::Receiver<Option<Instant>>,
    validity: Duration,
) -> Result<()> {
    let rpc = Duration::from_secs_f64(config.rpc_timeout_seconds);
    let channel = Endpoint::from_shared(config.endpoint.clone())?
        .connect_timeout(rpc)
        .timeout(rpc)
        .connect()
        .await?;
    let mut client = HandControlClient::new(channel);
    let mut leases = Vec::new();
    let operation: Result<()> = async {
        let status = client.get_status(Empty {}).await?.into_inner();
        ensure!(
            status.lease_timeout_seconds.is_finite()
                && status.lease_timeout_seconds
                    > config.renew_period_seconds + 2. * config.rpc_timeout_seconds,
            "hand lease timeout is too short for tracking renewal"
        );
        let sides: Vec<_> = status.hands.iter().map(|h| h.side).collect();
        ensure!(
            !sides.is_empty()
                && sides
                    .iter()
                    .all(|s| [Side::Left as i32, Side::Right as i32].contains(s))
                && sides
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == sides.len(),
            "invalid enabled hand list"
        );
        for side in sides {
            if !fresh(desired, validity) {
                return Ok(());
            }
            let lease = client.acquire(Hand { side }).await?.into_inner();
            ensure!(
                lease.side == side && !lease.token.is_empty(),
                "invalid hand tracking lease"
            );
            leases.push(lease);
        }
        for lease in &leases {
            if !fresh(desired, validity) {
                return Ok(());
            }
            client
                .set_command(Command {
                    side: lease.side,
                    token: lease.token.clone(),
                    source: Source::Tracking as i32,
                    target: None,
                    max_raw_rate_per_second: None,
                })
                .await?;
        }
        eprintln!("Hand tracking enabled for {} hand(s)", leases.len());
        let period = Duration::from_secs_f64(config.renew_period_seconds);
        while wait(desired, validity, tokio::time::Instant::now() + period).await {
            for lease in &leases {
                if !fresh(desired, validity) {
                    return Ok(());
                }
                let status = client.keep_alive(lease.clone()).await?.into_inner();
                let hand = status
                    .hands
                    .iter()
                    .find(|h| h.side == lease.side)
                    .context("missing leased tracking hand")?;
                ensure!(
                    hand.owned
                        && hand.source == Source::Tracking as i32
                        && hand.feedback_fresh
                        && hand.transport_fault.is_empty(),
                    "hand tracking lost ownership or CAN feedback: {} {}",
                    hand.suspension_reason,
                    hand.transport_fault
                );
            }
        }
        Ok(())
    }
    .await;
    let mut errors = Vec::new();
    for lease in leases {
        if let Err(error) = client.release(lease).await {
            errors.push(error.to_string());
        }
    }
    operation?;
    ensure!(
        errors.is_empty(),
        "hand tracking release failed: {errors:?}"
    );
    Ok(())
}
