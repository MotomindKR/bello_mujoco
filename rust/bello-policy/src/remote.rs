//! Latest operator intent, neutral rearming, and latched remote abort.
use anyhow::{Context, Result, ensure};
use bello_remote_api::{
    WatchCommandRequest, WatchCommandResponse,
    remote_command_service_client::RemoteCommandServiceClient,
};
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::watch;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub endpoint: String,
    pub staleness_ms: u64,
}

#[derive(Clone, Default)]
struct Snapshot {
    axes: [f32; 3],
    received: Option<Instant>,
    abort: bool,
}

#[derive(Default)]
struct Gate {
    previous: Option<(u64, u64, Instant)>,
    armed: bool,
    abort: bool,
}

impl Gate {
    fn receive(
        &mut self,
        message: WatchCommandResponse,
        now: Instant,
        stale: Duration,
    ) -> Result<Snapshot> {
        let axes = [
            message.normalized_forward,
            message.normalized_lateral,
            message.normalized_yaw,
        ];
        ensure!(
            axes.iter()
                .all(|x| x.is_finite() && (-1. ..=1.).contains(x)),
            "remote axes are nonfinite or outside [-1, 1]"
        );
        ensure!(
            (message.controller_connected && message.intent_armed) || axes == [0.; 3],
            "unarmed/disconnected remote supplied nonzero intent"
        );
        if let Some((sequence, stamp, received)) = self.previous {
            ensure!(
                message.sequence > sequence && message.sample_monotonic_ns > stamp,
                "remote sequence/timestamp did not increase"
            );
            if now.duration_since(received) > stale {
                self.armed = false;
            }
        }
        if !message.controller_connected || !message.intent_armed {
            self.armed = false;
        }
        if message.controller_connected && message.intent_armed && axes == [0.; 3] {
            self.armed = true;
        }
        self.previous = Some((message.sequence, message.sample_monotonic_ns, now));
        self.abort |= message.abort_active;
        Ok(Snapshot {
            axes: if self.armed {
                axes.map(|x| x as f32)
            } else {
                [0.; 3]
            },
            received: Some(now),
            abort: self.abort,
        })
    }
    fn disconnected(&mut self) -> Snapshot {
        self.previous = None;
        self.armed = false;
        Snapshot {
            abort: self.abort,
            ..Default::default()
        }
    }
}

pub struct Source {
    latest: watch::Receiver<Result<Snapshot, String>>,
    task: tokio::task::JoinHandle<()>,
    stale: Duration,
}

impl Source {
    pub async fn connect(config: &Config, timeout: Duration) -> Result<Self> {
        ensure!(
            config.staleness_ms > 0 && config.staleness_ms < 60_000,
            "invalid remote staleness limit"
        );
        let endpoint = tonic::transport::Endpoint::from_shared(config.endpoint.clone())?
            .connect_timeout(timeout);
        let stale = Duration::from_millis(config.staleness_ms);
        let (tx, latest) = watch::channel(Ok(Snapshot::default()));
        let task = tokio::spawn(async move {
            let mut gate = Gate::default();
            let mut backoff = Duration::from_millis(100);
            loop {
                let connection = async {
                    let channel = endpoint.connect().await?;
                    RemoteCommandServiceClient::new(channel)
                        .watch_command(WatchCommandRequest {})
                        .await
                        .map(|r| r.into_inner())
                        .map_err(anyhow::Error::from)
                };
                if let Ok(mut stream) = connection.await {
                    while let Ok(Some(message)) = stream.message().await {
                        match gate.receive(message, Instant::now(), stale) {
                            Ok(sample) => {
                                let _ = tx.send_replace(Ok(sample));
                                backoff = Duration::from_millis(100);
                            }
                            Err(error) => {
                                let _ = tx.send_replace(Err(error.to_string()));
                                return;
                            }
                        }
                    }
                }
                let _ = tx.send_replace(Ok(gate.disconnected()));
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(1));
            }
        });
        let mut source = Self {
            latest,
            task,
            stale,
        };
        tokio::time::timeout(timeout, async {
            loop {
                source.sample()?;
                if source
                    .latest
                    .borrow()
                    .as_ref()
                    .is_ok_and(|s| s.received.is_some_and(|t| t.elapsed() <= stale))
                {
                    return Ok::<_, anyhow::Error>(());
                }
                source
                    .latest
                    .changed()
                    .await
                    .context("remote task exited")?;
            }
        })
        .await
        .context("remote readiness timed out")??;
        Ok(source)
    }

    /// A stale/disconnected source requests standing; abort remains latched.
    pub fn sample(&self) -> Result<[f32; 3]> {
        let latest = self.latest.borrow();
        let sample = latest
            .as_ref()
            .map_err(|message| anyhow::anyhow!("invalid remote stream: {message}"))?;
        ensure!(!sample.abort, "operator requested remote abort");
        ensure!(!self.task.is_finished(), "remote task exited");
        Ok(
            if sample.received.is_some_and(|t| t.elapsed() <= self.stale) {
                sample.axes
            } else {
                [0.; 3]
            },
        )
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn message(sequence: u64, forward: f64) -> WatchCommandResponse {
        WatchCommandResponse {
            sequence,
            sample_monotonic_ns: sequence,
            controller_connected: true,
            intent_armed: true,
            normalized_forward: forward,
            ..Default::default()
        }
    }
    #[test]
    fn neutral_is_required_after_connect_staleness_and_reconnect() {
        let mut gate = Gate::default();
        let now = Instant::now();
        let stale = Duration::from_millis(200);
        assert_eq!(
            gate.receive(message(1, 0.5), now, stale).unwrap().axes,
            [0.; 3]
        );
        gate.receive(message(2, 0.), now, stale).unwrap();
        assert_eq!(
            gate.receive(message(3, 0.5), now, stale).unwrap().axes,
            [0.5, 0., 0.]
        );
        assert_eq!(
            gate.receive(message(4, 0.5), now + stale * 2, stale)
                .unwrap()
                .axes,
            [0.; 3]
        );
        gate.disconnected();
        assert_eq!(
            gate.receive(message(1, 0.5), now + stale * 3, stale)
                .unwrap()
                .axes,
            [0.; 3]
        );
    }
    #[test]
    fn invalid_live_messages_fail_and_abort_survives_reconnect() {
        let mut gate = Gate::default();
        let now = Instant::now();
        let stale = Duration::from_millis(200);
        assert!(gate.receive(message(1, f64::NAN), now, stale).is_err());
        gate.receive(message(1, 0.), now, stale).unwrap();
        assert!(gate.receive(message(1, 0.), now, stale).is_err());
        let mut abort = message(2, 0.);
        abort.abort_active = true;
        assert!(gate.receive(abort, now, stale).unwrap().abort);
        assert!(gate.disconnected().abort);
    }
}
