//! Diagnostic storage must not block motor control.
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};

const QUEUED_RECORDS: usize = 32;

pub(crate) struct PolicyLog {
    sender: SyncSender<Vec<u8>>,
    dropped: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
}

impl PolicyLog {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Self::with_writer(file, QUEUED_RECORDS)
    }

    fn with_writer(mut writer: impl Write + Send + 'static, capacity: usize) -> io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let worker_dropped = dropped.clone();
        let worker_failed = failed.clone();
        std::thread::Builder::new()
            .name("policy-log".into())
            .spawn(move || {
                let result = (|| -> io::Result<()> {
                    for record in receiver {
                        writer.write_all(&record)?;
                        let count = worker_dropped.swap(0, Ordering::Relaxed);
                        if count != 0 {
                            writeln!(
                                writer,
                                "{{\"event\":\"log_records_dropped\",\"count\":{count}}}"
                            )?;
                            eprintln!("Policy log dropped {count} records while storage was busy.");
                        }
                        writer.flush()?;
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    worker_failed.store(true, Ordering::Relaxed);
                    eprintln!("Policy logging stopped: {error}. Motor control continues.");
                }
            })?;
        Ok(Self {
            sender,
            dropped,
            failed,
        })
    }

    pub fn record(&self, value: &impl Serialize) -> serde_json::Result<()> {
        if self.failed.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        match self.sender.try_send(bytes) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.failed.store(true, Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    struct BlockedWriter {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        finished: mpsc::Sender<()>,
        bytes: Arc<Mutex<Vec<u8>>>,
        blocked: bool,
    }

    impl Write for BlockedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.blocked {
                self.blocked = true;
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            self.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for BlockedWriter {
        fn drop(&mut self) {
            let _ = self.finished.send(());
        }
    }

    #[test]
    fn stalled_storage_cannot_block_records_and_overflow_is_reported() {
        let (entered, wait_entered) = mpsc::channel();
        let (release, wait_release) = mpsc::channel();
        let (finished, wait_finished) = mpsc::channel();
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let log = PolicyLog::with_writer(
            BlockedWriter {
                entered,
                release: wait_release,
                finished,
                bytes: bytes.clone(),
                blocked: false,
            },
            2,
        )
        .unwrap();
        log.record(&serde_json::json!({"sample": 0})).unwrap();
        wait_entered.recv_timeout(Duration::from_secs(2)).unwrap();
        let (done, wait_done) = mpsc::channel();
        std::thread::spawn(move || {
            for i in 1..=10 {
                log.record(&serde_json::json!({"sample": i})).unwrap();
            }
            drop(log);
            done.send(()).unwrap();
        });
        // The writer remains blocked until after all submissions and Drop finish.
        let submitted = wait_done.recv_timeout(Duration::from_secs(2));
        release.send(()).unwrap();
        submitted.unwrap();
        wait_finished.recv_timeout(Duration::from_secs(2)).unwrap();
        let records = String::from_utf8(bytes.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            records
                .iter()
                .filter_map(|r| r["sample"].as_u64())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            records
                .iter()
                .filter_map(|r| r["count"].as_u64())
                .sum::<u64>(),
            8
        );
    }

    struct FailedWriter;
    impl Write for FailedWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("disk full"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn disk_failure_does_not_fail_control() {
        let log = PolicyLog::with_writer(FailedWriter, 2).unwrap();
        log.record(&serde_json::json!({"sample": 0})).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !log.failed.load(Ordering::Relaxed) {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        log.record(&serde_json::json!({"sample": 1})).unwrap();
    }
}
