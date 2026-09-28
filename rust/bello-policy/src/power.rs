//! Best-effort motor-rail telemetry for policy logs, independent of control timing.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use bello_hardware_client::{Robot, raw::TelemetryFilter, raw::proto};
use serde::Serialize;
use tokio::sync::watch;

#[derive(Clone, Debug, Serialize)]
pub struct MotorRail {
    pub voltage_v: f64,
    pub current_a: f64,
    pub sample_age_ms: f64,
    pub stale: bool,
    pub sequence: u8,
    pub connection_generation: u64,
}

#[derive(Clone)]
struct Reading {
    value: MotorRail,
    received: Instant,
    stale_timeout: Duration,
}

impl Reading {
    fn decode(board: &proto::PowerBoardTelemetry, timeout: Duration) -> Option<Self> {
        if board.status.as_ref()?.state != proto::DeviceState::Online as i32 {
            return None;
        }
        let mut samples = board.samples.iter().filter_map(|sample| {
            if let Some(proto::power_board_sample::Reading::MotorRail(rail)) = &sample.reading {
                Some((sample, rail))
            } else {
                None
            }
        });
        let (sample, rail) = samples.next()?;
        if samples.next().is_some() || !rail.voltage_v.is_finite() || !rail.current_a.is_finite() {
            return None;
        }
        Some(Self {
            value: MotorRail {
                voltage_v: rail.voltage_v,
                current_a: rail.current_a,
                sample_age_ms: sample.sample_age_ms as f64,
                stale: sample.stale,
                sequence: sample.sequence.try_into().ok()?,
                connection_generation: board.connection_generation,
            },
            received: Instant::now(),
            stale_timeout: timeout,
        })
    }

    fn snapshot(&self, now: Instant) -> MotorRail {
        let mut value = self.value.clone();
        value.sample_age_ms += now.duration_since(self.received).as_secs_f64() * 1000.;
        value.stale |= value.sample_age_ms >= self.stale_timeout.as_secs_f64() * 1000.;
        value
    }
}

type Latest = BTreeMap<String, Option<Reading>>;

pub struct Source {
    latest: watch::Receiver<Latest>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Source {
    pub fn start(robot: &Robot, timeout: Duration) -> Self {
        let boards: BTreeMap<_, _> = robot
            .power_boards()
            .map(|board| (board.name.clone(), board.stale_timeout))
            .collect();
        let empty: Latest = boards.keys().map(|name| (name.clone(), None)).collect();
        let (tx, latest) = watch::channel(empty.clone());
        let client = robot.raw_client();
        let task = (!boards.is_empty()).then(|| {
            tokio::spawn(async move {
                let filter = TelemetryFilter::default()
                    .include_power_boards(true)
                    .selected_power_boards(boards.keys().cloned());
                loop {
                    if let Ok(Ok(mut stream)) =
                        tokio::time::timeout(timeout, client.watch(filter.clone())).await
                    {
                        while let Ok(Some(event)) = stream.message().await {
                            if let proto::watch_telemetry_response::Payload::PowerBoard(board) =
                                event.payload
                                && let Some(stale_timeout) = boards.get(&board.board_name)
                            {
                                let reading = Reading::decode(&board, *stale_timeout);
                                tx.send_modify(|values| {
                                    values.insert(board.board_name, reading);
                                });
                            }
                        }
                    }
                    tx.send_replace(empty.clone());
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            })
        });
        Self { latest, task }
    }

    pub fn snapshot(&self) -> BTreeMap<String, Option<MotorRail>> {
        let latest = self.latest.borrow();
        let now = Instant::now();
        latest
            .iter()
            .map(|(board, reading)| {
                (
                    board.clone(),
                    reading.as_ref().map(|reading| reading.snapshot(now)),
                )
            })
            .collect()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> proto::PowerBoardTelemetry {
        proto::PowerBoardTelemetry {
            board_name: "supply".into(),
            connection_generation: 7,
            status: Some(proto::DeviceStatus {
                state: proto::DeviceState::Online as i32,
                ..Default::default()
            }),
            samples: vec![proto::PowerBoardSample {
                sequence: 23,
                sample_age_ms: 250,
                stale: false,
                reading: Some(proto::power_board_sample::Reading::MotorRail(
                    proto::PowerVoltageCurrent {
                        voltage_v: 48.5,
                        current_a: -1.25,
                    },
                )),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn logged_values_preserve_units_sign_and_freshness() {
        let reading = Reading::decode(&board(), Duration::from_secs(3)).unwrap();
        let fresh = reading.snapshot(reading.received + Duration::from_millis(750));
        let json = serde_json::to_value(&fresh).unwrap();
        assert_eq!(json["voltage_v"], 48.5);
        assert_eq!(json["current_a"], -1.25);
        assert_eq!(json["sample_age_ms"], 1000.);
        assert_eq!(json["stale"], false);
        assert_eq!(json["sequence"], 23);
        assert_eq!(json["connection_generation"], 7);
        assert!(
            reading
                .snapshot(reading.received + Duration::from_millis(2750))
                .stale
        );
        let mut value = board();
        value.samples[0].stale = true;
        let reading = Reading::decode(&value, Duration::from_secs(3)).unwrap();
        assert!(reading.snapshot(reading.received).stale);
    }

    #[test]
    fn missing_offline_and_invalid_samples_are_unknown() {
        let timeout = Duration::from_secs(3);
        let mut value = board();
        value.samples.clear();
        assert!(Reading::decode(&value, timeout).is_none());
        value = board();
        value.samples[0].reading = Some(proto::power_board_sample::Reading::Battery(
            proto::PowerBattery {
                voltage_v: 52.,
                current_a: 1.,
                soc_percent: None,
            },
        ));
        assert!(Reading::decode(&value, timeout).is_none());
        value = board();
        value.status = None;
        assert!(Reading::decode(&value, timeout).is_none());
        value.status = Some(proto::DeviceStatus::default());
        assert!(Reading::decode(&value, timeout).is_none());
        value = board();
        value.samples.push(value.samples[0].clone());
        assert!(Reading::decode(&value, timeout).is_none());
        value = board();
        value.samples[0].sequence = 256;
        assert!(Reading::decode(&value, timeout).is_none());
        for (voltage_v, current_a) in [(f64::NAN, 1.), (48., f64::INFINITY)] {
            value = board();
            value.samples[0].reading = Some(proto::power_board_sample::Reading::MotorRail(
                proto::PowerVoltageCurrent {
                    voltage_v,
                    current_a,
                },
            ));
            assert!(Reading::decode(&value, timeout).is_none());
        }
    }

    #[test]
    fn snapshots_distinguish_unknown_zero_and_disconnected_boards() {
        let mut value = board();
        value.samples[0].reading = Some(proto::power_board_sample::Reading::MotorRail(
            proto::PowerVoltageCurrent {
                voltage_v: 0.,
                current_a: 0.,
            },
        ));
        let empty = BTreeMap::from([("supply".into(), None), ("other".into(), None)]);
        let (tx, latest) = watch::channel(empty.clone());
        let source = Source { latest, task: None };
        assert!(source.snapshot().values().all(Option::is_none));
        tx.send_modify(|values| {
            values.insert(
                "supply".into(),
                Reading::decode(&value, Duration::from_secs(3)),
            );
        });
        let json = serde_json::to_value(source.snapshot()).unwrap();
        assert_eq!(json["supply"]["voltage_v"], 0.);
        assert!(json["other"].is_null());
        tx.send_replace(empty);
        assert!(source.snapshot().values().all(Option::is_none));
    }
}
