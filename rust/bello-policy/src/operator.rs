//! Runner-owned operator gate and single-client intent stream.

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use futures_util::Stream;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Request, Response, Status, transport::Server};

pub mod proto {
    tonic::include_proto!("bello.operator.v1");
}
use proto::operator_service_server::{OperatorService, OperatorServiceServer};
use proto::{Mode, OperatorInput, OperatorStatus};

const STALE: Duration = Duration::from_millis(250);
const NEUTRAL: f64 = 0.05;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestMode {
    Hold,
    Stand,
    Locomotion,
    Motion(String),
    Tracking,
    Sit,
    ConfirmSit,
    Abort,
    SetTracking(bool),
}

pub struct Snapshot {
    pub lying_start: bool,
    pub axes: [f32; 3],
    pub lost: bool,
    pub request: Option<RequestMode>,
}

struct State {
    status: OperatorStatus,
    last_sequence: Option<(u64, u64)>,
    received: Option<Instant>,
    axes: [f32; 3],
    neutral_required: bool,
    hold_since: Option<Instant>,
    confirm_since: Option<Instant>,
    abort_since: Option<Instant>,
    stop_consumed: bool,
    previous_toggle_tracking: bool,
    pending: Option<RequestMode>,
    hold_consumed: bool,
    confirm_consumed: bool,
    aborted: bool,
    clear_requested: bool,
}

impl State {
    fn new(motions: Vec<String>) -> Result<Self> {
        ensure!(
            motions.iter().all(|name| !name.is_empty()
                && name != "Stand"
                && name != "Locomotion"
                && name != "Motion"
                && name != "Tracking"
                && name != "Sit")
                && motions
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == motions.len(),
            "motion names must be unique and must not be reserved modes"
        );
        let mut modes = vec!["Locomotion".to_owned()];
        if !motions.is_empty() {
            modes.push("Motion".into());
        }
        Ok(Self {
            status: OperatorStatus {
                mode: Mode::Disarmed as i32,
                selected: "Locomotion".into(),
                startups: vec!["Crane".into()],
                selected_startup: "Crane".into(),
                active_motion: String::new(),
                phase: "disarmed".into(),
                progress: 0.,
                fault: String::new(),
                controller_connected: false,
                neutral_required: true,
                modes,
                selected_motion: motions.first().cloned().unwrap_or_default(),
                motions,
                can_clear_fault: false,
                recovery_error: String::new(),
                clear_fault_sequence: 0,
                velocity_command: None,
                hold_progress: 0.,
                confirm_progress: 0.,
                abort_progress: 0.,
                tracking_available: false,
                tracking_enabled: false,
                tracking_live: false,
                input_protocol_version: 3,
                ..OperatorStatus::default()
            },
            last_sequence: None,
            received: None,
            axes: [0.; 3],
            neutral_required: true,
            hold_since: None,
            confirm_since: None,
            abort_since: None,
            stop_consumed: false,
            previous_toggle_tracking: false,
            pending: None,
            hold_consumed: false,
            confirm_consumed: false,
            aborted: false,
            clear_requested: false,
        })
    }

    fn disconnect(&mut self) {
        self.received = None;
        self.axes = [0.; 3];
        self.neutral_required = true;
        self.status.neutral_required = true;
        self.status.controller_connected = false;
        self.hold_since = None;
        self.confirm_since = None;
        self.abort_since = None;
        self.status.hold_progress = 0.;
        self.status.confirm_progress = 0.;
        self.status.abort_progress = 0.;
        self.previous_toggle_tracking = false;
        self.pending = None;
        self.hold_consumed = false;
        self.confirm_consumed = false;
    }

    fn request_clear(&mut self, sequence: u64) {
        self.status.clear_fault_sequence = sequence;
        self.clear_requested = true;
        self.aborted = false;
        self.status.can_clear_fault = false;
        self.status.recovery_error.clear();
        self.status.phase = "clearing fault".into();
    }

    fn receive(&mut self, sample: OperatorInput, now: Instant) -> Result<()> {
        ensure!(
            sample.input_protocol_version == 3,
            "console input protocol version 3 required"
        );
        let axes = [sample.forward, sample.lateral, sample.yaw];
        ensure!(
            axes.iter()
                .all(|v| v.is_finite() && (-1. ..=1.).contains(v)),
            "invalid movement intent axes"
        );
        ensure!(
            (-1..=1).contains(&sample.mode_step) && (-1..=1).contains(&sample.motion_step),
            "invalid selection step"
        );
        ensure!(
            [
                sample.request_hold,
                sample.confirm_startup,
                sample.confirm_fault_recovery,
                !sample.request_mode.is_empty(),
                sample.confirm_sit,
                sample.abort,
            ]
            .into_iter()
            .filter(|active| *active)
            .count()
                <= 1,
            "conflicting operator actions"
        );
        if let Some((sequence, stamp)) = self.last_sequence {
            ensure!(
                sample.sequence > sequence && sample.sample_monotonic_ns > stamp,
                "operator sequence or timestamp did not increase"
            );
        }
        if self
            .received
            .is_some_and(|at| now.duration_since(at) > STALE)
        {
            self.disconnect();
        }
        self.last_sequence = Some((sample.sequence, sample.sample_monotonic_ns));
        self.received = Some(now);
        self.status.controller_connected = sample.connected;
        if sample.clear_fault {
            self.status.clear_fault_sequence = sample.sequence;
            if !self.status.can_clear_fault {
                self.status.recovery_error = "Wait for fault cleanup before clearing.".into();
            } else if neutral_input(&sample) {
                self.request_clear(sample.sequence);
            } else {
                self.status.recovery_error =
                    "Release all controller inputs before clearing fault.".into();
            }
        }
        // Abort works without a connected gamepad. A held abort may clear Fault
        // only after cleanup and a fresh neutral sample following the first stop.
        if sample.abort {
            let idle = self.status.mode == Mode::Fault as i32;
            if self.stop_consumed {
                self.axes = [0.; 3];
                return Ok(());
            }
            if idle
                && (!self.status.can_clear_fault
                    || (sample.connected && self.neutral_required)
                    || !sample.abort_other_controls_neutral
                    || !neutral_except_action(&sample))
            {
                self.abort_since = None;
                self.status.abort_progress = 0.;
                self.axes = [0.; 3];
                return Ok(());
            }
            let since = *self.abort_since.get_or_insert(now);
            self.status.abort_progress = now.duration_since(since).as_secs_f64().min(1.);
            self.axes = if idle || !sample.connected || self.neutral_required {
                [0.; 3]
            } else {
                axes.map(|v| if v.abs() <= NEUTRAL { 0. } else { v as f32 })
            };
            self.hold_since = None;
            self.confirm_since = None;
            self.status.hold_progress = 0.;
            self.status.confirm_progress = 0.;
            self.confirm_consumed = true;
            self.pending = None;
            if now.duration_since(since) >= Duration::from_secs(1) {
                self.axes = [0.; 3];
                self.stop_consumed = true;
                if idle {
                    self.request_clear(sample.sequence);
                } else {
                    self.aborted = true;
                }
            }
            return Ok(());
        }
        self.abort_since = None;
        self.stop_consumed = false;
        self.status.abort_progress = 0.;
        if let Some(selection) = &sample.selection {
            // Choices work without a gamepad, but never start control.
            self.confirm_since = None;
            self.status.confirm_progress = 0.;
            self.confirm_consumed = confirm_requested(&sample);
            self.pending = None;
            if !self.aborted
                && self.status.mode == Mode::Disarmed as i32
                && self.status.phase == "disarmed"
                && let Some(proto::selection::Choice::Startup(name)) = &selection.choice
                && self.status.startups.contains(name)
            {
                self.status.selected_startup.clone_from(name);
                self.hold_since = None;
                self.hold_consumed = sample.request_hold;
                self.status.hold_progress = 0.;
            }
            if !self.aborted && self.status.mode == Mode::Stand as i32 {
                match &selection.choice {
                    Some(proto::selection::Choice::Mode(name))
                        if self.status.modes.contains(name) =>
                    {
                        self.status.selected = name.clone();
                    }
                    Some(proto::selection::Choice::Motion(name))
                        if self.status.selected == "Motion"
                            && self.status.motions.contains(name) =>
                    {
                        self.status.selected_motion = name.clone();
                    }
                    _ => {}
                }
            }
        }
        if !sample.connected {
            self.disconnect();
            return Ok(());
        }
        if self.neutral_required {
            if neutral_input(&sample) {
                self.neutral_required = false;
                self.status.neutral_required = false;
            } else {
                self.axes = [0.; 3];
                return Ok(());
            }
        }
        self.axes = axes.map(|v| if v.abs() <= NEUTRAL { 0. } else { v as f32 });
        if self.status.mode == Mode::Fault as i32 {
            if !sample.confirm_fault_recovery {
                self.confirm_consumed = false;
            }
            let ready = self.status.can_clear_fault
                && sample.confirm_fault_recovery
                && !self.confirm_consumed
                && sample.selection.is_none()
                && neutral_except_confirm(&sample);
            if sample.confirm_fault_recovery && !ready {
                self.confirm_consumed = true;
            }
            self.confirm_since = ready.then(|| self.confirm_since.unwrap_or(now));
            self.status.confirm_progress = self
                .confirm_since
                .map_or(0., |start| now.duration_since(start).as_secs_f64().min(1.));
            if self
                .confirm_since
                .is_some_and(|start| now.duration_since(start) >= Duration::from_secs(1))
            {
                self.request_clear(sample.sequence);
                self.confirm_since = None;
                self.confirm_consumed = true;
                self.status.confirm_progress = 0.;
            }
            return Ok(());
        }
        if self.aborted {
            return Ok(());
        }
        let toggle_tracking = sample.toggle_tracking && !self.previous_toggle_tracking;
        self.previous_toggle_tracking = sample.toggle_tracking;
        let requested_tracking = match sample.selection.as_ref().and_then(|s| s.choice.as_ref()) {
            Some(proto::selection::Choice::TrackingEnabled(enabled)) => Some(*enabled),
            _ => toggle_tracking.then_some(!self.status.tracking_enabled),
        };
        if let Some(enabled) = requested_tracking
            && self.status.mode == Mode::Locomotion as i32
            && self.status.tracking_available
        {
            self.pending = Some(RequestMode::SetTracking(enabled));
            self.confirm_since = None;
            self.confirm_consumed = confirm_requested(&sample);
            self.status.confirm_progress = 0.;
            return Ok(());
        }
        if sample.selection.is_some() {
            return Ok(());
        }
        if self.status.mode == Mode::Disarmed as i32 {
            if self.status.phase != "disarmed" {
                return Ok(());
            }
            if sample.mode_step != 0 {
                cycle_selection(
                    &self.status.startups,
                    &mut self.status.selected_startup,
                    sample.mode_step as isize,
                );
                self.hold_since = None;
                self.status.hold_progress = 0.;
                self.hold_consumed = sample.request_hold;
            }
            if sample.navigating && sample.request_hold {
                self.hold_consumed = true;
            }
            if !self.status.fault.is_empty() {
                self.hold_since = None;
                self.status.hold_progress = 0.;
                return Ok(());
            }
            if !sample.request_hold {
                self.hold_consumed = false;
            }
            self.hold_since = if sample.request_hold
                && !self.hold_consumed
                && sample.mode_step == 0
                && sample.motion_step == 0
                && !sample.navigating
                && axes.iter().all(|v| v.abs() <= NEUTRAL)
            {
                Some(self.hold_since.unwrap_or(now))
            } else {
                None
            };
            self.status.hold_progress = self
                .hold_since
                .map_or(if self.hold_consumed { 1. } else { 0. }, |start| {
                    (now.duration_since(start).as_secs_f64() / 2.).min(1.)
                });
            if self
                .hold_since
                .is_some_and(|start| now.duration_since(start) >= Duration::from_secs(2))
            {
                self.pending = Some(RequestMode::Hold);
                self.hold_since = None;
                self.hold_consumed = true;
            }
            return Ok(());
        }
        if self.status.mode == Mode::Stand as i32 {
            if sample.mode_step != 0 {
                cycle_selection(
                    &self.status.modes,
                    &mut self.status.selected,
                    sample.mode_step as isize,
                );
            } else if self.status.selected == "Motion" && sample.motion_step != 0 {
                cycle_selection(
                    &self.status.motions,
                    &mut self.status.selected_motion,
                    sample.motion_step as isize,
                );
            }
        }
        if sample.navigating || sample.mode_step != 0 || sample.motion_step != 0 {
            self.confirm_since = None;
            self.status.confirm_progress = 0.;
            self.confirm_consumed = confirm_requested(&sample);
            return Ok(());
        }
        let requested = if sample.confirm_startup {
            match Mode::try_from(self.status.mode) {
                Ok(Mode::Hold)
                    if self.status.phase == "holding"
                        || (self.status.startup_ready
                            && matches!(
                                self.status.phase.as_str(),
                                "awaiting lift" | "upright"
                            )) =>
                {
                    Some(RequestMode::Stand)
                }
                _ => None,
            }
        } else if sample.confirm_sit {
            (self.status.mode == Mode::Sit as i32 && self.status.startup_ready)
                .then_some(RequestMode::ConfirmSit)
        } else if !sample.request_mode.is_empty() {
            match Mode::try_from(self.status.mode) {
                Ok(Mode::Stand) if self.status.selected == sample.request_mode => {
                    match sample.request_mode.as_str() {
                        "Locomotion" => Some(RequestMode::Locomotion),
                        "Motion" => Some(RequestMode::Motion(self.status.selected_motion.clone())),
                        "Tracking" if self.status.sonic_ready => Some(RequestMode::Tracking),
                        "Sit" => Some(RequestMode::Sit),
                        _ => None,
                    }
                }
                Ok(Mode::Locomotion | Mode::Tracking) if sample.request_mode == "Stand" => {
                    Some(RequestMode::Stand)
                }
                _ => None,
            }
        } else {
            None
        };
        if !confirm_requested(&sample) {
            self.confirm_consumed = false;
        }
        self.confirm_since = if requested.is_some()
            && !self.confirm_consumed
            && axes.iter().all(|v| v.abs() <= NEUTRAL)
        {
            Some(self.confirm_since.unwrap_or(now))
        } else {
            None
        };
        self.status.confirm_progress = self.confirm_since.map_or(
            if self.confirm_consumed && confirm_requested(&sample) {
                self.status.confirm_progress
            } else {
                0.
            },
            |start| now.duration_since(start).as_secs_f64().min(1.),
        );
        if self
            .confirm_since
            .is_some_and(|start| now.duration_since(start) >= Duration::from_secs(1))
        {
            self.pending = requested;
            self.confirm_since = None;
            self.confirm_consumed = true;
        }
        Ok(())
    }
}

fn confirm_requested(sample: &OperatorInput) -> bool {
    sample.confirm_startup
        || sample.confirm_sit
        || sample.confirm_fault_recovery
        || !sample.request_mode.is_empty()
}

fn neutral_except_confirm(sample: &OperatorInput) -> bool {
    [sample.forward, sample.lateral, sample.yaw]
        .iter()
        .all(|v| v.abs() <= NEUTRAL)
        && !sample.request_hold
        && !sample.confirm_startup
        && !sample.confirm_sit
        && sample.request_mode.is_empty()
        && !sample.abort
        && !sample.toggle_tracking
        && sample.mode_step == 0
        && sample.motion_step == 0
        && !sample.navigating
}

fn neutral_input(sample: &OperatorInput) -> bool {
    sample.controls_neutral
        && [sample.forward, sample.lateral, sample.yaw]
            .iter()
            .all(|v| v.abs() <= NEUTRAL)
        && !sample.request_hold
        && !confirm_requested(sample)
        && !sample.abort
        && !sample.toggle_tracking
        && sample.mode_step == 0
        && sample.motion_step == 0
        && !sample.navigating
        && sample.selection.is_none()
}

fn neutral_except_action(sample: &OperatorInput) -> bool {
    [sample.forward, sample.lateral, sample.yaw]
        .iter()
        .all(|v| v.abs() <= NEUTRAL)
        && !sample.request_hold
        && !confirm_requested(sample)
        && !sample.toggle_tracking
        && sample.mode_step == 0
        && sample.motion_step == 0
        && !sample.navigating
        && sample.selection.is_none()
}

fn cycle_selection(choices: &[String], selected: &mut String, direction: isize) {
    if choices.is_empty() {
        return;
    }
    let index = choices
        .iter()
        .position(|choice| choice == selected)
        .unwrap_or(0);
    *selected =
        choices[(index as isize + direction).rem_euclid(choices.len() as isize) as usize].clone();
}

pub struct Source {
    state: Arc<Mutex<State>>,
    shutdown: watch::Sender<bool>,
    pub listen_addr: SocketAddr,
}

impl Source {
    pub fn bind(listen: SocketAddr, motions: Vec<String>) -> Result<Self> {
        let state = Arc::new(Mutex::new(State::new(motions)?));
        let listener = std::net::TcpListener::bind(listen)?;
        listener.set_nonblocking(true)?;
        let listen_addr = listener.local_addr()?;
        let listener = tokio::net::TcpListener::from_std(listener)?;
        let (shutdown, mut shutdown_rx) = watch::channel(false);
        let service = Service {
            state: state.clone(),
            active: Arc::new(AtomicBool::new(false)),
            shutdown: shutdown_rx.clone(),
        };
        let failure_state = state.clone();
        tokio::spawn(async move {
            let result = Server::builder()
                .add_service(OperatorServiceServer::new(service).max_decoding_message_size(4096))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    let _ = shutdown_rx.changed().await;
                })
                .await;
            if let Err(error) = result {
                eprintln!("Operator service unavailable: {error}");
                failure_state.lock().unwrap().disconnect();
            }
        });
        Ok(Self {
            state,
            shutdown,
            listen_addr,
        })
    }

    pub fn snapshot(&self) -> Snapshot {
        let mut state = self.state.lock().unwrap();
        let stale = state.received.is_none_or(|at| at.elapsed() > STALE);
        if stale && state.status.controller_connected {
            state.disconnect();
        }
        let lost = stale || state.neutral_required;
        let request = if state.aborted {
            Some(RequestMode::Abort)
        } else {
            state.pending.take()
        };
        Snapshot {
            lying_start: state.status.selected_startup == "Back-down",
            axes: if lost { [0.; 3] } else { state.axes },
            lost,
            request,
        }
    }

    pub fn set_mode(&self, mode: Mode, phase: &str, active_motion: &str, progress: f64) {
        let mut state = self.state.lock().unwrap();
        if state.status.mode != mode as i32 || state.status.phase != phase {
            state.hold_since = None;
            state.confirm_since = None;
            state.status.hold_progress = 0.;
            state.status.confirm_progress = 0.;
            state.confirm_consumed = true;
            state.pending = None;
        }
        if !matches!(mode, Mode::Hold | Mode::Sit) {
            state.status.startup_ready = false;
        }
        if mode != Mode::Locomotion {
            state.status.velocity_command = None;
        }
        state.status.mode = mode as i32;
        state.status.phase = phase.into();
        state.status.active_motion = active_motion.into();
        state.status.progress = progress.clamp(0., 1.);
    }

    pub fn set_startups(&self, lying: bool) {
        let mut state = self.state.lock().unwrap();
        state.status.startups = vec!["Crane".into()];
        state.status.modes.retain(|mode| mode != "Sit");
        if lying {
            state.status.startups.push("Back-down".into());
            state.status.modes.push("Sit".into());
        }
    }

    pub fn set_startup_ready(&self, ready: bool) {
        let mut state = self.state.lock().unwrap();
        state.status.startup_ready = ready;
        if !ready {
            state.confirm_since = None;
            state.status.confirm_progress = 0.;
        }
    }

    pub fn set_startup_error(&self, message: &str) {
        let mut state = self.state.lock().unwrap();
        if state.status.mode != Mode::Disarmed as i32 {
            return;
        }
        if state.status.fault.is_empty() != message.is_empty() {
            state.disconnect();
            eprintln!(
                "{}",
                if message.is_empty() {
                    "Startup ready; release inputs before the Hold chord."
                } else {
                    message
                }
            );
        }
        state.status.fault = message.into();
    }

    pub fn set_velocity_command(&self, [forward, right, yaw]: [f32; 3]) {
        let mut state = self.state.lock().unwrap();
        state.status.velocity_command =
            (state.status.mode == Mode::Locomotion as i32).then_some(proto::VelocityCommand {
                forward_mps: forward as f64,
                right_mps: right as f64,
                yaw_rps: yaw as f64,
            });
    }

    pub fn set_tracking(&self, available: bool, enabled: bool, live: bool) {
        let mut state = self.state.lock().unwrap();
        state.status.tracking_available = available;
        state.status.tracking_enabled = enabled;
        state.status.tracking_live = live;
    }

    pub fn set_sonic(&self, status: &crate::tracking::Status) {
        let mut state = self.state.lock().unwrap();
        if !state.status.modes.iter().any(|mode| mode == "Tracking") {
            state.status.modes.push("Tracking".into());
        }
        if !status.ready && state.status.selected == "Tracking" {
            state.confirm_since = None;
            state.status.confirm_progress = 0.;
        }
        state.status.sonic_ready = status.ready;
        state.status.sonic_reference_age_ms = status.age_ms;
        state.status.sonic_buffered_ms = status.buffered_ms;
        state.status.sonic_delay_ms = status.delay_ms;
        state.status.sonic_status.clone_from(&status.message);
    }

    pub fn fault(&self, message: &str) {
        let mut state = self.state.lock().unwrap();
        state.disconnect();
        state.status.velocity_command = None;
        state.status.startup_ready = false;
        state.status.mode = Mode::Fault as i32;
        state.status.phase = "fault".into();
        state.status.fault = message.into();
        state.status.can_clear_fault = false;
    }

    pub fn allow_fault_clear(&self, error: &str) {
        let mut state = self.state.lock().unwrap();
        state.status.can_clear_fault = state.status.mode == Mode::Fault as i32;
        state.status.phase = "fault".into();
        state.status.recovery_error = error.into();
    }

    pub fn take_fault_clear(&self) -> bool {
        std::mem::take(&mut self.state.lock().unwrap().clear_requested)
    }

    pub fn disarm(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.aborted {
            return false;
        }
        state.disconnect();
        state.aborted = false;
        state.clear_requested = false;
        state.status.startup_ready = false;
        state.status.mode = Mode::Disarmed as i32;
        state.status.phase = "disarmed".into();
        state.status.selected = "Locomotion".into();
        state.status.active_motion.clear();
        state.status.progress = 0.;
        state.status.fault.clear();
        state.status.recovery_error.clear();
        state.status.can_clear_fault = false;
        true
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.shutdown.send_replace(true);
    }
}

#[derive(Clone)]
struct Service {
    state: Arc<Mutex<State>>,
    active: Arc<AtomicBool>,
    shutdown: watch::Receiver<bool>,
}

struct OperatorLease {
    active: Arc<AtomicBool>,
    state: Arc<Mutex<State>>,
}
impl Drop for OperatorLease {
    fn drop(&mut self) {
        self.state.lock().unwrap().disconnect();
        self.active.store(false, Ordering::SeqCst);
    }
}

#[tonic::async_trait]
impl OperatorService for Service {
    type ExchangeStream = Pin<Box<dyn Stream<Item = Result<OperatorStatus, Status>> + Send>>;

    async fn exchange(
        &self,
        request: Request<tonic::Streaming<OperatorInput>>,
    ) -> Result<Response<Self::ExchangeStream>, Status> {
        if self
            .active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(Status::resource_exhausted(
                "Runner is in use by another console. Close that console to release its operator lease.",
            ));
        }
        let lease = OperatorLease {
            active: self.active.clone(),
            state: self.state.clone(),
        };
        {
            let mut state = self.state.lock().unwrap();
            state.disconnect();
            state.last_sequence = None;
            state.status.clear_fault_sequence = 0;
        }
        let mut input = request.into_inner();
        let (tx, rx) = mpsc::channel(1);
        let shared = self.state.clone();
        let mut shutdown = self.shutdown.clone();
        tokio::spawn(async move {
            let _lease = lease;
            let mut heartbeat = tokio::time::interval(Duration::from_millis(50));
            heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut expires = tokio::time::Instant::now() + STALE;
            loop {
                tokio::select! {
                    biased;
                    _ = shutdown.changed() => break,
                    _ = tx.closed() => break,
                    _ = tokio::time::sleep_until(expires) => {
                        let _ = tx.try_send(Err(Status::deadline_exceeded("Operator lease expired: no operator inputs for 250 ms.")));
                        break;
                    },
                    message = input.message() => match message {
                        Ok(Some(sample)) => {
                            let result = shared.lock().unwrap().receive(sample, Instant::now());
                            if let Err(error) = result {
                                let _ = tx.try_send(Err(Status::invalid_argument(error.to_string())));
                                break;
                            }
                            expires = tokio::time::Instant::now() + STALE;
                        }
                        _ => break,
                    },
                    _ = heartbeat.tick() => {
                        let status = shared.lock().unwrap().status.clone();
                        // A slow display must never block receiving an abort.
                        let _ = tx.try_send(Ok(status));
                    }
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Existing controller scenarios are translated at the test boundary. The
    // production runner only receives semantic OperatorInput messages.
    #[derive(Clone, Default)]
    struct GamepadSample {
        sequence: u64,
        sample_monotonic_ns: u64,
        connected: bool,
        forward: f64,
        lateral: f64,
        yaw: f64,
        back: bool,
        start: bool,
        a: bool,
        y: bool,
        l1: bool,
        r1: bool,
        l2: bool,
        r2: bool,
        dpad_up: bool,
        dpad_down: bool,
        dpad_left: bool,
        dpad_right: bool,
        clear_fault: bool,
        abort: bool,
        selection: Option<proto::Selection>,
        input_protocol_version: u32,
    }

    struct Controller {
        now: Instant,
        sequence: u64,
        left: bool,
        right: bool,
        up: bool,
        down: bool,
        tracking: bool,
    }
    impl Controller {
        fn new() -> Self {
            Self {
                now: Instant::now(),
                sequence: 0,
                left: false,
                right: false,
                up: false,
                down: false,
                tracking: false,
            }
        }
        fn map(&mut self, sample: GamepadSample, status: &OperatorStatus) -> OperatorInput {
            let axes_neutral = [sample.forward, sample.lateral, sample.yaw]
                .iter()
                .all(|v| v.abs() <= NEUTRAL);
            let others = sample.back
                || sample.start
                || sample.a
                || sample.y
                || sample.l1
                || sample.r1
                || sample.dpad_up
                || sample.dpad_down
                || sample.dpad_left
                || sample.dpad_right;
            let abort = sample.abort || (sample.l2 && sample.r2);
            let mode_step = if sample.dpad_right && !sample.dpad_left && !self.right {
                1
            } else if sample.dpad_left && !sample.dpad_right && !self.left {
                -1
            } else {
                0
            };
            let motion_step = if sample.dpad_down && !sample.dpad_up && !self.down {
                1
            } else if sample.dpad_up && !sample.dpad_down && !self.up {
                -1
            } else {
                0
            };
            let toggle_tracking = sample.y && !self.tracking;
            self.left = sample.dpad_left;
            self.right = sample.dpad_right;
            self.up = sample.dpad_up;
            self.down = sample.dpad_down;
            self.tracking = sample.y;
            let confirm = sample.a
                && !sample.back
                && !sample.start
                && !sample.y
                && !sample.l1
                && !sample.r1
                && !sample.l2
                && !sample.r2;
            OperatorInput {
                sequence: sample.sequence,
                sample_monotonic_ns: sample.sample_monotonic_ns,
                connected: sample.connected,
                forward: sample.forward,
                lateral: sample.lateral,
                yaw: sample.yaw,
                controls_neutral: axes_neutral && !others && !sample.l2 && !sample.r2,
                abort_other_controls_neutral: axes_neutral && !others,
                request_hold: sample.back
                    && sample.start
                    && !abort
                    && !sample.a
                    && !sample.y
                    && !sample.l1
                    && !sample.r1,
                confirm_startup: confirm && status.mode == Mode::Hold as i32,
                confirm_fault_recovery: confirm && status.mode == Mode::Fault as i32,
                confirm_sit: confirm && status.mode == Mode::Sit as i32,
                request_mode: if confirm {
                    match Mode::try_from(status.mode) {
                        Ok(Mode::Stand) => status.selected.clone(),
                        Ok(Mode::Locomotion | Mode::Tracking) => "Stand".into(),
                        _ => String::new(),
                    }
                } else {
                    String::new()
                },
                abort,
                clear_fault: sample.clear_fault,
                selection: sample.selection,
                mode_step,
                motion_step,
                navigating: sample.dpad_left
                    || sample.dpad_right
                    || sample.dpad_up
                    || sample.dpad_down,
                toggle_tracking,
                input_protocol_version: sample.input_protocol_version,
                ..Default::default()
            }
        }
        fn send(&mut self, state: &mut State, sample: GamepadSample, milliseconds: u64) {
            for _ in 0..milliseconds.div_ceil(20) {
                self.now += Duration::from_millis(20);
                self.sequence += 1;
                let mut sample = sample.clone();
                sample.sequence = self.sequence;
                sample.sample_monotonic_ns = self.sequence;
                sample.connected = true;
                sample.input_protocol_version = 3;
                let input = self.map(sample, &state.status);
                state.receive(input, self.now).unwrap();
            }
        }
    }
    fn fixture() -> (State, Controller) {
        let mut state = State::new(vec!["Wave".into(), "Bow".into()]).unwrap();
        let mut controller = Controller::new();
        controller.send(&mut state, GamepadSample::default(), 20);
        (state, controller)
    }
    #[test]
    fn semantic_abort_hold_clears_fault_without_gamepad_after_release() {
        let mut state = State::new(vec![]).unwrap();
        let started = Instant::now();
        let mut sequence = 0;
        let mut send = |state: &mut State, abort: bool, count: u64| {
            for _ in 0..count {
                sequence += 1;
                state
                    .receive(
                        OperatorInput {
                            sequence,
                            sample_monotonic_ns: sequence,
                            controls_neutral: !abort,
                            abort_other_controls_neutral: true,
                            abort,
                            input_protocol_version: 3,
                            ..Default::default()
                        },
                        started + Duration::from_millis(sequence * 20),
                    )
                    .unwrap();
            }
        };
        send(&mut state, true, 52);
        assert!(state.aborted);
        state.status.mode = Mode::Fault as i32;
        state.status.can_clear_fault = true;
        send(&mut state, true, 52);
        assert!(!state.clear_requested);
        send(&mut state, false, 1);
        send(&mut state, true, 52);
        assert!(state.clear_requested);
        assert!(!state.aborted);
    }
    #[test]
    fn sit_requires_stand_and_fresh_neutral_confirmations() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        state.status.modes.push("Sit".into());
        state.status.selected = "Sit".into();
        let a = GamepadSample {
            a: true,
            ..Default::default()
        };
        controller.send(&mut state, a.clone(), 900);
        assert!(state.pending.is_none());
        controller.send(&mut state, a.clone(), 140);
        assert_eq!(state.pending.take(), Some(RequestMode::Sit));
        state.status.mode = Mode::Sit as i32;
        state.status.startup_ready = true;
        controller.send(&mut state, a.clone(), 1200);
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(
            &mut state,
            GamepadSample {
                forward: -0.5,
                ..a.clone()
            },
            1200,
        );
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, a.clone(), 1040);
        assert_eq!(state.pending.take(), Some(RequestMode::ConfirmSit));
        state.status.startup_ready = false;
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, a, 1200);
        assert!(state.pending.is_none());
    }
    #[test]
    fn startup_selection_requires_a_new_chord_and_balance_requires_readiness() {
        let (mut state, mut controller) = fixture();
        state.status.startups.push("Back-down".into());
        controller.send(
            &mut state,
            GamepadSample {
                back: true,
                start: true,
                ..Default::default()
            },
            1000,
        );
        controller.send(
            &mut state,
            GamepadSample {
                back: true,
                start: true,
                dpad_right: true,
                ..Default::default()
            },
            1200,
        );
        assert_eq!(state.status.selected_startup, "Back-down");
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(
            &mut state,
            GamepadSample {
                back: true,
                start: true,
                ..Default::default()
            },
            2040,
        );
        assert_eq!(state.pending.take(), Some(RequestMode::Hold));
        state.status.mode = Mode::Hold as i32;
        state.status.phase = "awaiting lift".into();
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1200,
        );
        assert!(state.pending.is_none());
        state.status.startup_ready = true;
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1020,
        );
        assert_eq!(state.pending.take(), Some(RequestMode::Stand));
        state.status.phase = "straightening".into();
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1200,
        );
        assert!(state.pending.is_none());
        // Configuration selection cannot change during Hold.
        controller.send(
            &mut state,
            GamepadSample {
                selection: Some(proto::Selection {
                    choice: Some(proto::selection::Choice::Startup("Crane".into())),
                }),
                ..Default::default()
            },
            20,
        );
        assert_eq!(state.status.selected_startup, "Back-down");
    }

    #[test]
    fn tracking_needs_a_ready_window_and_a_full_confirm_hold() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        state.status.modes.push("Tracking".into());
        state.status.selected = "Tracking".into();
        let a = GamepadSample {
            a: true,
            ..Default::default()
        };
        controller.send(&mut state, a.clone(), 1500);
        assert!(state.pending.is_none());
        assert_eq!(state.status.confirm_progress, 0.);
        state.status.sonic_ready = true;
        controller.send(&mut state, a.clone(), 800);
        assert!(state.pending.is_none());
        controller.send(&mut state, a.clone(), 240);
        assert_eq!(state.pending.take(), Some(RequestMode::Tracking));
        state.status.mode = Mode::Tracking as i32;
        controller.send(&mut state, a.clone(), 1400);
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, a, 1040);
        assert_eq!(state.pending.take(), Some(RequestMode::Stand));
    }
    #[tokio::test]
    async fn fault_clear_requires_completed_cleanup_and_neutral_then_disarms() {
        let source = Source::bind("127.0.0.1:0".parse().unwrap(), vec!["Wave".into()]).unwrap();
        let mut controller = Controller::new();
        let clear = GamepadSample {
            clear_fault: true,
            ..Default::default()
        };
        source.fault("CAN failure");
        controller.send(&mut source.state.lock().unwrap(), clear.clone(), 20);
        assert!(!source.take_fault_clear());
        source.allow_fault_clear("");
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample {
                a: true,
                ..clear.clone()
            },
            20,
        );
        assert!(!source.take_fault_clear());
        assert!(
            !source
                .state
                .lock()
                .unwrap()
                .status
                .recovery_error
                .is_empty()
        );
        controller.send(&mut source.state.lock().unwrap(), clear.clone(), 20);
        assert!(source.take_fault_clear());
        controller.send(&mut source.state.lock().unwrap(), clear, 20);
        assert!(!source.take_fault_clear());
        source.state.lock().unwrap().aborted = true;
        assert!(!source.disarm());
        source.allow_fault_clear("");
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample {
                clear_fault: true,
                ..Default::default()
            },
            20,
        );
        assert!(source.take_fault_clear());
        assert!(source.disarm());
        assert!(source.snapshot().lost);
        assert_eq!(source.snapshot().request, None);
        let chord = GamepadSample {
            back: true,
            start: true,
            ..Default::default()
        };
        controller.send(&mut source.state.lock().unwrap(), chord.clone(), 2200);
        assert_eq!(source.snapshot().request, None);
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample::default(),
            20,
        );
        controller.send(&mut source.state.lock().unwrap(), chord, 2200);
        assert_eq!(source.snapshot().request, Some(RequestMode::Hold));
    }
    #[tokio::test]
    async fn trigger_abort_then_idle_requires_release_and_allows_rearming() {
        let source = Source::bind("127.0.0.1:0".parse().unwrap(), vec![]).unwrap();
        let mut controller = Controller::new();
        let triggers = GamepadSample {
            l2: true,
            r2: true,
            ..Default::default()
        };
        controller.send(&mut source.state.lock().unwrap(), triggers.clone(), 1040);
        assert_eq!(source.snapshot().request, Some(RequestMode::Abort));
        source.fault("Operator abort");
        source.allow_fault_clear("");
        controller.send(&mut source.state.lock().unwrap(), triggers.clone(), 2200);
        assert!(!source.take_fault_clear());
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample::default(),
            20,
        );
        controller.send(&mut source.state.lock().unwrap(), triggers.clone(), 800);
        assert!(!source.take_fault_clear());
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample::default(),
            20,
        );
        controller.send(&mut source.state.lock().unwrap(), triggers.clone(), 1040);
        assert!(source.take_fault_clear());
        assert_ne!(source.snapshot().request, Some(RequestMode::Abort));
        assert!(source.disarm());
        controller.send(&mut source.state.lock().unwrap(), triggers, 2200);
        assert_ne!(source.snapshot().request, Some(RequestMode::Abort));
        assert!(!source.take_fault_clear());
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample::default(),
            20,
        );
        controller.send(
            &mut source.state.lock().unwrap(),
            GamepadSample {
                back: true,
                start: true,
                ..Default::default()
            },
            2040,
        );
        assert_eq!(source.snapshot().request, Some(RequestMode::Hold));
    }

    #[test]
    fn idle_hold_requires_cleanup_neutral_sticks_and_continuous_samples() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Fault as i32;
        let triggers = GamepadSample {
            l2: true,
            r2: true,
            ..Default::default()
        };
        controller.send(&mut state, triggers.clone(), 1040);
        assert!(!state.clear_requested);
        state.status.can_clear_fault = true;
        controller.send(
            &mut state,
            GamepadSample {
                forward: 0.5,
                ..triggers.clone()
            },
            1040,
        );
        assert!(!state.clear_requested);
        controller.send(&mut state, triggers.clone(), 800);
        controller.now += STALE + Duration::from_millis(1);
        controller.send(&mut state, triggers.clone(), 1040);
        assert!(!state.clear_requested);
        assert_eq!(state.status.abort_progress, 0.);
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, triggers, 1040);
        assert!(state.clear_requested);
    }

    #[test]
    fn fault_a_hold_requires_cleanup_release_and_neutral_input() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Fault as i32;
        state.aborted = true;
        let a = GamepadSample {
            a: true,
            ..Default::default()
        };
        controller.send(&mut state, a.clone(), 1200);
        assert!(!state.clear_requested);
        state.status.can_clear_fault = true;
        controller.send(&mut state, a.clone(), 1200);
        assert!(!state.clear_requested);
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(
            &mut state,
            GamepadSample {
                forward: 0.5,
                ..a.clone()
            },
            1200,
        );
        assert!(!state.clear_requested);
        controller.send(&mut state, a.clone(), 1200);
        assert!(!state.clear_requested);
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, a.clone(), 1040);
        assert!(state.clear_requested);
        assert!(!state.aborted);
        assert!(!state.status.can_clear_fault);
        state.clear_requested = false;
        state.status.can_clear_fault = true;
        controller.send(&mut state, a, 1200);
        assert!(!state.clear_requested);
    }

    #[test]
    fn chord_is_continuous_and_fires_once_until_release() {
        let (mut state, mut controller) = fixture();
        let chord = GamepadSample {
            back: true,
            start: true,
            ..Default::default()
        };
        controller.send(&mut state, chord.clone(), 1900);
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, chord.clone(), 2000);
        assert!(state.pending.is_none());
        controller.send(&mut state, chord.clone(), 40);
        assert_eq!(state.pending.take(), Some(RequestMode::Hold));
        controller.send(&mut state, chord, 2100);
        assert!(state.pending.is_none());
    }
    #[test]
    fn console_selection_requires_stand_and_known_choices() {
        let (mut state, mut controller) = fixture();
        let select = GamepadSample {
            selection: Some(proto::Selection {
                choice: Some(proto::selection::Choice::Mode("Motion".into())),
            }),
            ..Default::default()
        };
        for mode in [
            Mode::Disarmed,
            Mode::Hold,
            Mode::Locomotion,
            Mode::Motion,
            Mode::Fault,
        ] {
            state.status.mode = mode as i32;
            controller.send(&mut state, select.clone(), 20);
            assert_eq!(state.status.selected, "Locomotion");
            assert!(state.pending.is_none());
        }
        state.status.mode = Mode::Stand as i32;
        controller.send(&mut state, select, 20);
        assert_eq!(state.status.selected, "Motion");
        for name in ["Bow", "Unknown"] {
            controller.send(
                &mut state,
                GamepadSample {
                    selection: Some(proto::Selection {
                        choice: Some(proto::selection::Choice::Motion(name.into())),
                    }),
                    ..Default::default()
                },
                20,
            );
            assert_eq!(state.status.selected_motion, "Bow");
            assert!(state.pending.is_none());
        }
        for name in ["Stand", "Unknown"] {
            controller.send(
                &mut state,
                GamepadSample {
                    selection: Some(proto::Selection {
                        choice: Some(proto::selection::Choice::Mode(name.into())),
                    }),
                    ..Default::default()
                },
                20,
            );
            assert_eq!(state.status.selected, "Motion");
        }
    }
    #[test]
    fn console_selection_cancels_confirmation_until_release_and_full_hold() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        let a = GamepadSample {
            a: true,
            ..Default::default()
        };
        controller.send(&mut state, a.clone(), 800);
        controller.send(
            &mut state,
            GamepadSample {
                selection: Some(proto::Selection {
                    choice: Some(proto::selection::Choice::Mode("Locomotion".into())),
                }),
                ..a.clone()
            },
            20,
        );
        controller.send(&mut state, a.clone(), 1400);
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, a.clone(), 1000);
        assert!(state.pending.is_none());
        controller.send(&mut state, a, 40);
        assert_eq!(state.pending.take(), Some(RequestMode::Locomotion));
    }
    #[test]
    fn console_selection_without_gamepad_does_not_bypass_reconnect_neutrality() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        let input = controller.map(
            GamepadSample {
                sequence: 2,
                sample_monotonic_ns: 2,
                connected: false,
                input_protocol_version: 3,
                selection: Some(proto::Selection {
                    choice: Some(proto::selection::Choice::Mode("Locomotion".into())),
                }),
                ..Default::default()
            },
            &state.status,
        );
        state
            .receive(input, controller.now + Duration::from_millis(20))
            .unwrap();
        controller.sequence = 2;
        controller.now += Duration::from_millis(20);
        assert_eq!(state.status.selected, "Locomotion");
        assert!(state.neutral_required);
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1200,
        );
        assert!(state.pending.is_none());
        assert!(state.neutral_required);
    }
    #[test]
    fn abort_takes_priority_over_console_selection() {
        for abort in [
            GamepadSample {
                abort: true,
                ..Default::default()
            },
            GamepadSample {
                l2: true,
                r2: true,
                ..Default::default()
            },
        ] {
            let (mut state, mut controller) = fixture();
            state.status.mode = Mode::Stand as i32;
            controller.send(
                &mut state,
                GamepadSample {
                    selection: Some(proto::Selection {
                        choice: Some(proto::selection::Choice::Mode("Motion".into())),
                    }),
                    ..abort
                },
                1040,
            );
            assert!(state.aborted);
            assert_eq!(state.status.selected, "Locomotion");
        }
    }
    #[test]
    fn two_named_motions_are_selected_from_stand() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        let right = GamepadSample {
            dpad_right: true,
            ..Default::default()
        };
        controller.send(&mut state, right, 40);
        assert_eq!(state.status.selected, "Motion");
        controller.send(&mut state, GamepadSample::default(), 20);
        assert_eq!(state.status.selected_motion, "Wave");
        controller.send(
            &mut state,
            GamepadSample {
                dpad_down: true,
                ..Default::default()
            },
            200,
        );
        assert_eq!(state.status.selected_motion, "Bow");
        controller.send(&mut state, GamepadSample::default(), 20);
        let a = GamepadSample {
            a: true,
            ..Default::default()
        };
        controller.send(&mut state, a.clone(), 1400);
        assert_eq!(
            state.pending.take(),
            Some(RequestMode::Motion("Bow".into()))
        );
        state.status.mode = Mode::Motion as i32;
        controller.send(&mut state, a, 2000);
        assert!(state.pending.is_none());
    }
    #[test]
    fn motion_list_wraps_and_navigation_requires_fresh_confirmation() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        state.status.selected = "Motion".into();
        controller.send(
            &mut state,
            GamepadSample {
                dpad_up: true,
                ..Default::default()
            },
            200,
        );
        assert_eq!(state.status.selected_motion, "Bow");
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            800,
        );
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                dpad_down: true,
                ..Default::default()
            },
            40,
        );
        assert_eq!(state.status.selected_motion, "Wave");
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1200,
        );
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1040,
        );
        assert_eq!(
            state.pending.take(),
            Some(RequestMode::Motion("Wave".into()))
        );
    }

    #[test]
    fn motion_mode_requires_bundles_and_its_name_is_reserved() {
        assert_eq!(State::new(vec![]).unwrap().status.modes, ["Locomotion"]);
        assert!(State::new(vec!["Motion".into()]).is_err());
    }

    #[test]
    fn tracking_toggle_requires_locomotion_configuration_and_fresh_press() {
        let (mut state, mut controller) = fixture();
        let y = GamepadSample {
            y: true,
            ..Default::default()
        };
        state.status.mode = Mode::Stand as i32;
        state.status.tracking_available = true;
        controller.send(&mut state, y.clone(), 40);
        assert!(state.pending.is_none());
        state.status.mode = Mode::Locomotion as i32;
        controller.send(&mut state, y.clone(), 40);
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, y.clone(), 40);
        assert_eq!(state.pending.take(), Some(RequestMode::SetTracking(true)));
        controller.send(&mut state, y.clone(), 1000);
        assert!(state.pending.is_none());
        controller.send(&mut state, GamepadSample::default(), 20);
        state.status.tracking_available = false;
        controller.send(&mut state, y, 40);
        assert!(state.pending.is_none());
    }

    #[test]
    fn repeated_semantic_tracking_toggle_is_one_request() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Locomotion as i32;
        state.status.tracking_available = true;
        for _ in 0..2 {
            controller.sequence += 1;
            controller.now += Duration::from_millis(20);
            state
                .receive(
                    OperatorInput {
                        sequence: controller.sequence,
                        sample_monotonic_ns: controller.sequence,
                        connected: true,
                        toggle_tracking: true,
                        input_protocol_version: 3,
                        ..Default::default()
                    },
                    controller.now,
                )
                .unwrap();
            if controller.sequence == 2 {
                assert_eq!(state.pending.take(), Some(RequestMode::SetTracking(true)));
            } else {
                assert!(state.pending.is_none());
            }
        }
    }

    #[test]
    fn tracking_selection_requires_ready_controller_and_locomotion() {
        let (mut state, mut controller) = fixture();
        let selection = GamepadSample {
            selection: Some(proto::Selection {
                choice: Some(proto::selection::Choice::TrackingEnabled(true)),
            }),
            ..Default::default()
        };
        state.status.tracking_available = true;
        state.status.mode = Mode::Stand as i32;
        controller.send(&mut state, selection.clone(), 20);
        assert!(state.pending.is_none());
        state.status.mode = Mode::Locomotion as i32;
        controller.send(&mut state, selection.clone(), 20);
        assert_eq!(state.pending.take(), Some(RequestMode::SetTracking(true)));
        state.disconnect();
        controller.send(
            &mut state,
            GamepadSample {
                y: true,
                ..selection.clone()
            },
            20,
        );
        assert!(state.pending.is_none());
        state.status.tracking_available = false;
        controller.send(&mut state, selection, 20);
        assert!(state.pending.is_none());
    }

    #[test]
    fn selection_and_confirmation_are_rejected_outside_allowed_phases() {
        for (mode, phase) in [
            (Mode::Hold, "ramping"),
            (Mode::Motion, "motion"),
            (Mode::Fault, "fault"),
        ] {
            let (mut state, mut controller) = fixture();
            state.status.mode = mode as i32;
            state.status.phase = phase.into();
            controller.send(
                &mut state,
                GamepadSample {
                    dpad_right: true,
                    a: true,
                    ..Default::default()
                },
                2100,
            );
            assert_eq!(state.status.selected, "Locomotion");
            assert!(state.pending.is_none());
        }
    }

    #[test]
    fn console_abort_does_not_require_a_connected_or_neutral_gamepad() {
        let mut state = State::new(vec![]).unwrap();
        let start = Instant::now();
        for sequence in 1..=52 {
            state
                .receive(
                    OperatorInput {
                        sequence,
                        sample_monotonic_ns: sequence,
                        abort: true,
                        connected: false,
                        input_protocol_version: 3,
                        forward: 0.5,
                        ..Default::default()
                    },
                    start + Duration::from_millis(sequence * 20),
                )
                .unwrap();
        }
        assert!(state.aborted);
        assert_eq!(state.axes, [0.; 3]);
    }

    #[test]
    fn horizontal_selection_wraps_and_requires_release() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        let left = GamepadSample {
            dpad_left: true,
            ..Default::default()
        };
        let right = GamepadSample {
            dpad_right: true,
            ..Default::default()
        };
        controller.send(&mut state, left.clone(), 40);
        assert_eq!(state.status.selected, "Motion");
        controller.send(&mut state, left, 200);
        assert_eq!(state.status.selected, "Motion");
        controller.send(&mut state, GamepadSample::default(), 20);
        controller.send(&mut state, right.clone(), 40);
        assert_eq!(state.status.selected, "Locomotion");
        controller.send(&mut state, right, 200);
        assert_eq!(state.status.selected, "Locomotion");
        for sample in [
            GamepadSample {
                dpad_up: true,
                ..Default::default()
            },
            GamepadSample {
                dpad_down: true,
                ..Default::default()
            },
            GamepadSample {
                dpad_left: true,
                dpad_right: true,
                ..Default::default()
            },
        ] {
            controller.send(&mut state, GamepadSample::default(), 20);
            controller.send(&mut state, sample, 40);
            assert_eq!(state.status.selected, "Locomotion");
        }
    }

    #[test]
    fn horizontal_buttons_require_neutral_on_reconnect() {
        for sample in [
            GamepadSample {
                dpad_left: true,
                ..Default::default()
            },
            GamepadSample {
                dpad_right: true,
                ..Default::default()
            },
        ] {
            let (mut state, mut controller) = fixture();
            state.disconnect();
            controller.send(&mut state, sample, 40);
            assert!(state.status.neutral_required);
            controller.send(&mut state, GamepadSample::default(), 20);
            assert!(!state.status.neutral_required);
        }
    }

    #[test]
    fn neutral_input_cannot_become_a_movement_command() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Locomotion as i32;
        for value in [-NEUTRAL, -0.00874, 0., 0.00874, NEUTRAL] {
            controller.send(
                &mut state,
                GamepadSample {
                    forward: value,
                    lateral: value,
                    yaw: value,
                    ..Default::default()
                },
                20,
            );
            assert_eq!(state.axes, [0.; 3]);
        }
        controller.send(
            &mut state,
            GamepadSample {
                forward: 0.00874,
                lateral: -0.5,
                yaw: 0.25,
                ..Default::default()
            },
            20,
        );
        assert_eq!(state.axes, [0., -0.5, 0.25]);
        controller.send(&mut state, GamepadSample::default(), 20);
        assert_eq!(state.axes, [0.; 3]);
    }

    #[test]
    fn hold_progress_cancels_on_release_and_abort_timer_resets_after_sample_gap() {
        let (mut state, mut controller) = fixture();
        let entry = GamepadSample {
            back: true,
            start: true,
            ..Default::default()
        };
        controller.send(&mut state, entry, 1000);
        assert!((0.45..0.5).contains(&state.status.hold_progress));
        controller.send(&mut state, GamepadSample::default(), 20);
        assert_eq!(state.status.hold_progress, 0.);
        state.status.mode = Mode::Stand as i32;
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            500,
        );
        assert!((0.45..0.5).contains(&state.status.confirm_progress));
        controller.send(&mut state, GamepadSample::default(), 20);
        assert_eq!(state.status.confirm_progress, 0.);
        let abort = GamepadSample {
            l2: true,
            r2: true,
            ..Default::default()
        };
        controller.send(&mut state, abort.clone(), 800);
        controller.now += STALE + Duration::from_millis(1);
        controller.send(&mut state, abort.clone(), 500);
        assert!(!state.aborted);
        assert!((0.45..0.5).contains(&state.status.abort_progress));
        state.disconnect();
        assert_eq!(state.status.abort_progress, 0.);
        controller.send(&mut state, abort, 1000);
        assert!(!state.aborted);
    }
    #[test]
    fn abort_requires_both_triggers_continuously_for_one_second() {
        for mode in [
            Mode::Disarmed,
            Mode::Hold,
            Mode::Stand,
            Mode::Locomotion,
            Mode::Motion,
        ] {
            let (mut state, mut controller) = fixture();
            state.status.mode = mode as i32;
            for (l2, r2) in [(true, false), (false, true), (false, false)] {
                controller.send(
                    &mut state,
                    GamepadSample {
                        l2,
                        r2,
                        ..Default::default()
                    },
                    40,
                );
                assert!(!state.aborted);
            }
            controller.send(
                &mut state,
                GamepadSample {
                    l2: true,
                    r2: true,
                    ..Default::default()
                },
                1000,
            );
            assert!(!state.aborted);
            assert!(state.status.abort_progress > 0.9);
            controller.send(
                &mut state,
                GamepadSample {
                    l2: true,
                    ..Default::default()
                },
                20,
            );
            assert_eq!(state.status.abort_progress, 0.);
            controller.send(
                &mut state,
                GamepadSample {
                    l2: true,
                    r2: true,
                    ..Default::default()
                },
                1000,
            );
            assert!(!state.aborted);
            controller.send(
                &mut state,
                GamepadSample {
                    l2: true,
                    r2: true,
                    ..Default::default()
                },
                40,
            );
            assert_eq!(state.status.abort_progress, 1.);
            assert!(state.aborted);
            assert_eq!(state.axes, [0.; 3]);
        }
    }

    #[test]
    fn unfinished_abort_hold_keeps_locomotion_intent() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Locomotion as i32;
        let moving = GamepadSample {
            forward: 0.4,
            l2: true,
            r2: true,
            ..Default::default()
        };
        controller.send(&mut state, moving.clone(), 300);
        assert!(!state.aborted);
        assert_eq!(state.axes, [0.4, 0., 0.]);
        controller.send(
            &mut state,
            GamepadSample {
                forward: 0.4,
                ..Default::default()
            },
            20,
        );
        assert_eq!(state.status.abort_progress, 0.);
        controller.send(&mut state, moving, 1040);
        assert!(state.aborted);
        assert_eq!(state.axes, [0.; 3]);
    }

    #[test]
    fn held_bumpers_do_not_abort() {
        let (mut state, mut controller) = fixture();
        assert_eq!(state.status.input_protocol_version, 3);
        controller.send(
            &mut state,
            GamepadSample {
                l1: true,
                r1: true,
                ..Default::default()
            },
            1200,
        );
        assert!(!state.aborted);
        assert_eq!(state.status.abort_progress, 0.);
    }

    #[test]
    fn old_console_samples_are_rejected_before_control() {
        let mut state = State::new(vec![]).unwrap();
        let sample = OperatorInput {
            connected: true,
            sequence: 1,
            sample_monotonic_ns: 1,
            input_protocol_version: 2,
            ..Default::default()
        };
        assert!(
            state
                .receive(sample, Instant::now())
                .unwrap_err()
                .to_string()
                .contains("input protocol version 3 required")
        );
        assert!(!state.aborted);
        assert!(state.received.is_none());
    }

    #[test]
    fn either_trigger_prevents_reconnect_neutrality() {
        let mut state = State::new(vec![]).unwrap();
        let mut controller = Controller::new();
        for (l2, r2) in [(true, false), (false, true)] {
            controller.send(
                &mut state,
                GamepadSample {
                    l2,
                    r2,
                    ..Default::default()
                },
                20,
            );
            assert!(state.neutral_required);
            assert!(!state.aborted);
        }
        controller.send(&mut state, GamepadSample::default(), 20);
        assert!(!state.neutral_required);
    }

    #[test]
    fn abort_works_before_neutral_and_survives_disconnect() {
        let mut state = State::new(vec![]).unwrap();
        let mut controller = Controller::new();
        controller.send(
            &mut state,
            GamepadSample {
                l2: true,
                r2: true,
                ..Default::default()
            },
            1040,
        );
        assert!(state.aborted);
        state.disconnect();
        controller.send(&mut state, GamepadSample::default(), 20);
        assert!(state.aborted);
    }
    #[test]
    fn stale_gap_rejects_held_chord_and_requires_neutral() {
        let (mut state, mut controller) = fixture();
        let chord = GamepadSample {
            back: true,
            start: true,
            ..Default::default()
        };
        controller.send(&mut state, chord.clone(), 1000);
        controller.now += Duration::from_secs(2);
        controller.send(&mut state, chord, 2000);
        assert!(state.pending.is_none());
        assert!(state.neutral_required);
    }
    #[test]
    fn loss_clears_unconsumed_transitions_and_axes() {
        let (mut state, mut controller) = fixture();
        state.status.mode = Mode::Stand as i32;
        state.status.selected = "Motion".into();
        controller.send(
            &mut state,
            GamepadSample {
                a: true,
                ..Default::default()
            },
            1040,
        );
        assert!(state.pending.is_some());
        state.disconnect();
        assert!(state.pending.is_none());
        controller.send(
            &mut state,
            GamepadSample {
                forward: 1.,
                ..Default::default()
            },
            20,
        );
        assert_eq!(state.axes, [0.; 3]);
    }
    #[test]
    fn sequence_is_strict_even_across_controller_disconnect() {
        let (mut state, controller) = fixture();
        let invalid = OperatorInput {
            sequence: controller.sequence,
            sample_monotonic_ns: controller.sequence,
            connected: false,
            input_protocol_version: 3,
            ..Default::default()
        };
        assert!(state.receive(invalid, controller.now).is_err());
    }
    #[tokio::test]
    async fn disconnected_controller_samples_renew_console_lease_and_rejected_clients_cannot_abort()
    {
        use proto::operator_service_client::OperatorServiceClient;
        let source = Source::bind("127.0.0.1:0".parse().unwrap(), vec![]).unwrap();
        let endpoint = format!("http://{}", source.listen_addr);
        let mut owner = OperatorServiceClient::connect(endpoint.clone())
            .await
            .unwrap();
        let mut other = OperatorServiceClient::connect(endpoint).await.unwrap();
        let (tx, rx) = mpsc::channel::<OperatorInput>(4);
        let mut stream = owner
            .exchange(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        for sequence in 1..=8 {
            tx.send(OperatorInput {
                sequence,
                sample_monotonic_ns: sequence,
                connected: false,
                input_protocol_version: 3,
                ..Default::default()
            })
            .await
            .unwrap();
            assert!(stream.message().await.unwrap().is_some());
            let rejected = other
                .exchange(tokio_stream::iter([OperatorInput {
                    connected: true,
                    abort: true,
                    abort_other_controls_neutral: true,
                    sequence: 1,
                    sample_monotonic_ns: 1,
                    ..Default::default()
                }]))
                .await
                .unwrap_err();
            assert_eq!(rejected.code(), tonic::Code::ResourceExhausted);
            assert!(!source.state.lock().unwrap().aborted);
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        drop(stream);
        drop(tx);
        let replacement = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let (sender, receiver) = mpsc::channel::<OperatorInput>(4);
                match other.exchange(ReceiverStream::new(receiver)).await {
                    Ok(response) => break (sender, response.into_inner()),
                    Err(error) => assert_eq!(error.code(), tonic::Code::ResourceExhausted),
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let (_sender, mut replacement) = replacement;
        assert!(
            replacement
                .message()
                .await
                .unwrap()
                .unwrap()
                .neutral_required
        );
        assert!(source.snapshot().lost);
    }

    #[tokio::test]
    async fn only_one_stream_owns_the_session_and_silent_owner_expires() {
        use proto::operator_service_client::OperatorServiceClient;
        let source = Source::bind("127.0.0.1:0".parse().unwrap(), vec![]).unwrap();
        let mut client = OperatorServiceClient::connect(format!("http://{}", source.listen_addr))
            .await
            .unwrap();
        let (_tx, rx) = mpsc::channel::<OperatorInput>(4);
        let mut first = client
            .exchange(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        assert!(first.message().await.unwrap().unwrap().neutral_required);
        let (_tx2, rx2) = mpsc::channel::<OperatorInput>(4);
        let error = client.exchange(ReceiverStream::new(rx2)).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
        loop {
            match first.message().await {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
                    break;
                }
            }
        }
        let (_tx3, rx3) = mpsc::channel::<OperatorInput>(4);
        let mut third = client
            .exchange(ReceiverStream::new(rx3))
            .await
            .unwrap()
            .into_inner();
        assert!(third.message().await.unwrap().unwrap().neutral_required);
    }
}
