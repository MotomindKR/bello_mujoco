//! Read-only reference transport shared by upper-body following and SONIC.
use anyhow::Result;
use std::error::Error;
use std::time::{Duration, Instant};
use tokio::sync::watch;
pub mod proto {
    tonic::include_proto!("bello.reference.v1");
}
use proto::{
    GetReferenceRequest, GetReferenceResponse, reference_service_client::ReferenceServiceClient,
};

#[derive(Clone)]
pub struct Sample {
    pub message: GetReferenceResponse,
    pub requested: Instant,
}

pub struct Source {
    latest: watch::Receiver<Result<Option<Sample>, String>>,
    task: tokio::task::JoinHandle<()>,
}
pub(crate) fn connection_lost(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::Unavailable | tonic::Code::DeadlineExceeded
    ) || status
        .source()
        .is_some_and(|source| source.is::<tonic::transport::Error>())
}

impl Source {
    pub fn new(
        address: &str,
        robot: &str,
        rpc_timeout_seconds: f64,
        poll_period_seconds: f64,
    ) -> Result<Self> {
        let endpoint = tonic::transport::Endpoint::from_shared(address.to_owned())?
            .connect_timeout(Duration::from_secs_f64(rpc_timeout_seconds));
        let timeout = Duration::from_secs_f64(rpc_timeout_seconds);
        let period = Duration::from_secs_f64(poll_period_seconds);
        let request = GetReferenceRequest {
            robot: robot.to_owned(),
        };
        let (tx, latest) = watch::channel(Ok(None));
        let address = address.to_owned();
        let task = tokio::spawn(async move {
            let mut client = ReferenceServiceClient::new(endpoint.connect_lazy())
                .max_decoding_message_size(64 * 1024);
            let mut last_error = None;
            loop {
                let requested = Instant::now();
                let sample = match tokio::time::timeout(
                    timeout,
                    client.get_reference(request.clone()),
                )
                .await
                {
                    Ok(Ok(response)) => {
                        if last_error.take().is_some() {
                            eprintln!("Reference producer reconnected: {address}");
                        }
                        Ok(Some(Sample {
                            message: response.into_inner(),
                            requested,
                        }))
                    }
                    Err(_) => Ok(None),
                    Ok(Err(status)) if connection_lost(&status) => {
                        let message = status.to_string();
                        if last_error.as_ref() != Some(&message) {
                            eprintln!(
                                "Reference producer unavailable at {address}; retrying: {message}"
                            );
                            last_error = Some(message);
                        }
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
        Ok(Self { latest, task })
    }
    pub fn latest(&self) -> Result<Option<Sample>, String> {
        self.latest.borrow().clone()
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        self.task.abort();
    }
}
