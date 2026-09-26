//! Explicit operator lifecycle over bello-stack's lease-owning control sessions.
use crate::{
    hardware::Hardware,
    locomotion::{Feedback, Locomotion},
    remote,
};
use anyhow::{Context, Result, ensure};
use bello_hardware_client::{
    CommandFrame, ControlOptions, ControlSession, ControlState, CoordinateTarget, MotorGroup,
    raw::proto,
};
use futures_util::future::join_all;
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use serde::Deserialize;
use std::fs::{File, OpenOptions};
use std::io::{IsTerminal, Write};
use std::os::fd::{AsFd, AsRawFd};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub following: crate::following::Config,
    pub client_name: String,
    pub remote: remote::Config,
    pub command_validity_seconds: f64,
    pub acknowledgement_timeout_seconds: f64,
    pub rpc_timeout_seconds: f64,
    pub ramp_seconds: f64,
    pub gain_scale: f64,
    pub joystick_command_scale: f32,
    #[serde(default)]
    pub torso_command: [f32; 3],
    pub max_loop_lag_seconds: f64,
    pub max_duration_seconds: Option<f64>,
    pub max_nominal_error_radians: f64,
    pub max_coupled_nominal_error_radians: f64,
    pub policy_log_path: Option<PathBuf>,
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(!self.client_name.is_empty(), "missing control owner");
        for value in [
            self.command_validity_seconds,
            self.acknowledgement_timeout_seconds,
            self.rpc_timeout_seconds,
            self.ramp_seconds,
            self.gain_scale,
            self.max_loop_lag_seconds,
            self.max_nominal_error_radians,
            self.max_coupled_nominal_error_radians,
        ] {
            ensure!(
                value.is_finite() && value > 0. && value < 86400.,
                "invalid control timing/gain/limit"
            );
        }
        ensure!(
            self.acknowledgement_timeout_seconds > self.command_validity_seconds
                && self.gain_scale <= 1.,
            "invalid acknowledgement deadline/gain scale"
        );
        ensure!(
            self.max_duration_seconds
                .is_none_or(|s| s.is_finite() && s > 0. && s < 86400.),
            "invalid run duration"
        );
        Ok(())
    }
}

struct Operator {
    input: Vec<u8>,
}
impl Operator {
    fn new() -> Result<Self> {
        ensure!(
            std::io::stdin().is_terminal(),
            "armed control requires an interactive terminal"
        );
        Ok(Self { input: Vec::new() })
    }
    fn prompt(&mut self, text: &str) -> Result<()> {
        self.input.clear();
        writeln!(std::io::stderr(), "{text}")?;
        Ok(())
    }
    fn line(&mut self) -> Result<Option<String>> {
        let stdin = std::io::stdin();
        let mut fds = [PollFd::new(stdin.as_fd(), PollFlags::POLLIN)];
        while poll(&mut fds, PollTimeout::ZERO)? > 0 {
            let events = fds[0].revents().unwrap_or(PollFlags::empty());
            ensure!(
                !events.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL),
                "operator terminal failed"
            );
            if !events.intersects(PollFlags::POLLIN | PollFlags::POLLHUP) {
                break;
            }
            let mut byte = [0];
            // Raw fd avoids stdio prefetch making poll miss already-buffered characters.
            ensure!(
                nix::unistd::read(stdin.as_raw_fd(), &mut byte)? == 1,
                "operator terminal closed"
            );
            if byte[0] == b'\n' {
                return Ok(Some(String::from_utf8(std::mem::take(&mut self.input))?));
            }
            ensure!(self.input.len() < 32, "operator confirmation too long");
            if byte[0] != b'\r' {
                self.input.push(byte[0]);
            }
        }
        Ok(None)
    }
    async fn confirm(&mut self, expected: &str) -> Result<()> {
        loop {
            if let Some(line) = self.line()? {
                if expected == "IDLE" && line != expected {
                    eprintln!("Expected exact IDLE; controls remain under daemon safety.");
                    continue;
                }
                ensure!(line == expected, "expected exact {expected}");
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

fn frame(
    group: &MotorGroup,
    policy: &Locomotion,
    target: &[f64],
    scale: f64,
) -> Result<CommandFrame> {
    let metadata = &policy.policy.metadata;
    ensure!(
        target.len() == metadata.coordinates.len(),
        "motor target shape mismatch"
    );
    let mut builder = group.frame();
    for motor in group.motors() {
        let i = metadata
            .coordinates
            .iter()
            .position(|c| c == motor.coordinate())
            .context("missing policy coordinate")?;
        let limits = motor.limits();
        ensure!(
            target[i].is_finite()
                && target[i] >= limits.position_min()
                && target[i] <= limits.position_max(),
            "target for {} exceeds hardware limits",
            motor.name()
        );
        builder = builder.coordinate(
            motor.coordinate(),
            CoordinateTarget::position(target[i])
                .gains(metadata.kp[i] as f64 * scale, metadata.kd[i] as f64 * scale),
        )?;
    }
    Ok(builder.build()?)
}

fn nominal(policy: &Locomotion, sample: &Feedback, config: &Config) -> Result<()> {
    for (i, (measured, target)) in sample
        .motor_position
        .iter()
        .zip(&policy.policy.metadata.default_position)
        .enumerate()
    {
        let limit = policy.nominal_tolerance(
            i,
            config.max_nominal_error_radians,
            config.max_coupled_nominal_error_radians,
        );
        ensure!(
            (measured - *target as f64).abs() <= limit,
            "{} is too far from nominal pose",
            policy.policy.metadata.coordinates[i]
        );
    }
    Ok(())
}

async fn send(
    controls: &mut [ControlSession],
    policy: &Locomotion,
    target: &[f64],
    config: &Config,
    targets: &mut CommandTargets,
    limits: &crate::target_limits::TargetLimits,
    log: &mut Option<File>,
) -> Result<crate::target_limits::LimitedTarget> {
    let command = limits.constrain(target, &targets.previous)?;
    let frames = controls
        .iter()
        .map(|c| frame(c.group(), policy, &command.motor_target, config.gain_scale))
        .collect::<Result<Vec<_>>>()?;
    let results = join_all(controls.iter_mut().zip(frames).map(|(c, f)| c.send(f))).await;
    for result in results {
        if let Err(error) = result {
            if let Some(log) = log {
                serde_json::to_writer(
                    &mut *log,
                    &serde_json::json!({"event": "command_rejected", "command": command, "error": error.to_string()}),
                )?;
                log.write_all(b"\n")?;
                log.flush()?;
            }
            return Err(error.into());
        }
    }
    targets.previous.clone_from(&command.motor_target);
    if command.limited {
        targets.limited_count += 1;
        if targets.limited_count == 1 || targets.limited_count.is_multiple_of(50) {
            eprintln!(
                "Constrained motor/joint/step targets in {} commands; inspect policy log before commissioning.",
                targets.limited_count
            );
        }
        if let Some(log) = log {
            serde_json::to_writer(
                &mut *log,
                &serde_json::json!({"event": "target_limited", "command": command, "limited_count": targets.limited_count}),
            )?;
            log.write_all(b"\n")?;
            log.flush()?;
        }
    }
    Ok(command)
}

struct CommandTargets {
    previous: Vec<f64>,
    limited_count: u64,
}

async fn pace(deadline: &mut tokio::time::Instant, period: Duration, lag: Duration) -> Result<()> {
    *deadline += period;
    ensure!(
        tokio::time::Instant::now().saturating_duration_since(*deadline) < lag,
        "policy loop missed its deadline"
    );
    tokio::time::sleep_until(*deadline).await;
    Ok(())
}

pub async fn run(
    hardware: Hardware,
    mut policy: Locomotion,
    config: Config,
    mut motion: Option<crate::motion::live::Live>,
) -> Result<()> {
    config.validate()?;
    ensure!(
        config.command_validity_seconds > policy.policy.metadata.control_timestep,
        "command validity must exceed the policy period"
    );
    if let Some(motion) = &motion {
        ensure!(
            !config.following.enabled,
            "motion control cannot enable upper-body following"
        );
        ensure!(
            config.command_validity_seconds > motion.motion_period(),
            "command validity must exceed the motion period"
        );
    }
    let motion_interrupt = motion.as_ref().map(|motion| motion.interrupt.clone());
    let mut operator = Operator::new()?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    policy.remote_command([0.; 3], config.joystick_command_scale)?;
    policy.validate_torso_command(config.torso_command)?;
    ensure!(
        motion.is_none() || config.torso_command == [0.; 3],
        "motion playback requires a neutral standing torso command"
    );
    ensure!(
        !config.following.enabled || config.torso_command == [0.; 3],
        "body following owns torso commands; leave control.torso_command neutral"
    );
    let rpc = Duration::from_secs_f64(config.rpc_timeout_seconds);
    let remote = remote::Source::start(&config.remote, rpc)?;
    let mut following = if config.following.enabled {
        Some(crate::following::Following::new(
            config.following.clone(),
            &policy.policy,
        )?)
    } else {
        None
    };
    let mut log: Option<File> = config
        .policy_log_path
        .as_ref()
        .map(|p| OpenOptions::new().write(true).create_new(true).open(p))
        .transpose()?;
    let preflight = hardware.feedback(proto::GroupState::Idle).await?;
    let limits = hardware.target_limits.clone();
    let startup_margin = hardware.config.startup_feedback_margin_radians;
    limits.startup_target(&preflight.motor_position, startup_margin)?;
    if let Some(motion) = &motion {
        motion.check_feedback(&preflight)?;
    }
    nominal(&policy, &preflight, &config)?;
    // Warm up ONNX before acquiring control; discard its history before RUN.
    policy.step(&preflight)?;
    policy.reset();
    operator.prompt("Support the robot. Type HOLD to acquire control and ramp to nominal:")?;
    tokio::select! { r = operator.confirm("HOLD") => r?, _ = interrupt.recv() => anyhow::bail!("cancelled before arming"), _ = terminate.recv() => anyhow::bail!("cancelled before arming") }
    let mut controls = Vec::new();
    let operation = async {
        remote.sample()?;
        let initial = hardware.feedback(proto::GroupState::Idle).await?;
        if let Some(motion) = &motion {
            motion.check_feedback(&initial)?;
        }
        nominal(&policy, &initial, &config)?;
        let startup_target = limits.startup_target(&initial.motor_position, startup_margin)?;
        if startup_target != initial.motor_position {
            eprintln!(
                "Adjusted initial HOLD target inside command limits; measured feedback was within the configured feedback allowance."
            );
        }
        if let Some(log) = &mut log {
            serde_json::to_writer(
                &mut *log,
                &serde_json::json!({
                    "event": "startup_target", "measured_motor_position": initial.motor_position,
                    "motor_target": startup_target
                }),
            )?;
            log.write_all(b"\n")?;
            log.flush()?;
        }
        let mut targets = CommandTargets {
            previous: startup_target.clone(),
            limited_count: 0,
        };
        for group in &hardware.groups {
            let options = ControlOptions::new(&config.client_name)
                .command_validity(Duration::from_secs_f64(config.command_validity_seconds))
                .acknowledgement_timeout(Duration::from_secs_f64(
                    config.acknowledgement_timeout_seconds,
                ))
                .task_shutdown_timeout(rpc);
            controls.push(
                tokio::time::timeout(rpc, group.control_with(options))
                    .await
                    .context("lease acquisition timed out")??,
            );
        }
        let frames = controls
            .iter()
            .map(|c| frame(c.group(), &policy, &startup_target, config.gain_scale))
            .collect::<Result<Vec<_>>>()?;
        let results = join_all(
            controls
                .iter_mut()
                .zip(frames)
                .map(|(c, f)| tokio::time::timeout(rpc, c.start(f))),
        )
        .await;
        for result in results {
            result.context("motor start timed out")??;
        }
        let period = Duration::from_secs_f64(policy.policy.metadata.control_timestep);
        let lag = Duration::from_secs_f64(config.max_loop_lag_seconds);
        let target: Vec<f64> = policy
            .policy
            .metadata
            .default_position
            .iter()
            .map(|v| *v as f64)
            .collect();
        let steps = (config.ramp_seconds / period.as_secs_f64()).ceil() as usize;
        let mut deadline = tokio::time::Instant::now();
        for step in 1..=steps {
            remote.sample()?;
            let sample = hardware.feedback(proto::GroupState::Running).await?;
            if let Some(motion) = &motion {
                motion.check_feedback(&sample)?;
            }
            let ramp: Vec<f64> = startup_target
                .iter()
                .zip(&target)
                .map(|(a, b)| a + step as f64 / steps as f64 * (b - a))
                .collect();
            send(
                &mut controls,
                &policy,
                &ramp,
                &config,
                &mut targets,
                &limits,
                &mut log,
            )
            .await?;
            pace(&mut deadline, period, lag).await?;
        }
        operator.prompt(
            "Nominal ramp complete. Release support and settle the robot, then type RUN:",
        )?;
        loop {
            remote.sample()?;
            let sample = hardware.feedback(proto::GroupState::Running).await?;
            if let Some(motion) = &motion {
                motion.check_feedback(&sample)?;
            }
            nominal(&policy, &sample, &config)?;
            send(
                &mut controls,
                &policy,
                &target,
                &config,
                &mut targets,
                &limits,
                &mut log,
            )
            .await?;
            if let Some(line) = operator.line()? {
                ensure!(line == "RUN", "expected exact RUN");
                break;
            }
            pace(&mut deadline, period, lag).await?;
        }
        policy.reset();
        deadline = tokio::time::Instant::now();
        let started = Instant::now();
        let mut active_period = period;
        let mut torso_command = config.torso_command;
        if motion.is_some() {
            operator.prompt(
                "Standing policy is active with zero command. Type MOTION to begin the clip:",
            )?;
        } else if following.is_some() {
            operator.prompt("Locomotion active with body following; Ctrl-C stops.")?;
        } else {
            operator.prompt(
                "Locomotion active. Optional: TORSO <roll> <pitch> <yaw> in radians; Ctrl-C stops.",
            )?;
        }
        while config
            .max_duration_seconds
            .is_none_or(|s| started.elapsed().as_secs_f64() < s)
        {
            let max_tilt = motion
                .as_ref()
                .map_or(Some(hardware.config.max_tilt_radians), |motion| {
                    motion.absolute_tilt_limit(hardware.config.max_tilt_radians)
                });
            let mut sample = hardware
                .feedback_with_tilt(proto::GroupState::Running, max_tilt)
                .await?;
            if let Some(motion) = &mut motion {
                remote.sample()?;
                let request = operator.line()?;
                ensure!(
                    request.as_deref().is_none_or(|s| s == "MOTION"),
                    "expected exact MOTION"
                );
                let step = motion.step(
                    &mut policy,
                    &sample,
                    request.is_some(),
                    &targets.previous,
                    hardware.config.max_tilt_radians,
                )?;
                if step.period != active_period {
                    deadline = tokio::time::Instant::now();
                    active_period = step.period;
                }
                let target: Vec<f64> = step.target.iter().map(|v| *v as f64).collect();
                let command = send(
                    &mut controls,
                    &policy,
                    &target,
                    &config,
                    &mut targets,
                    motion.target_limits(),
                    &mut log,
                )
                .await?;
                if let Some(log) = &mut log {
                    serde_json::to_writer(
                        &mut *log,
                        &serde_json::json!({"elapsed_seconds": started.elapsed().as_secs_f64(), "feedback": sample, "motion": step.log, "command": command}),
                    )?;
                    log.write_all(b"\n")?;
                    log.flush()?;
                }
                pace(&mut deadline, active_period, lag).await?;
                continue;
            }
            sample.command =
                policy.remote_command(remote.sample()?, config.joystick_command_scale)?;
            if let Some(line) = operator.line()? {
                let fields: Vec<_> = line.split_whitespace().collect();
                let requested = (|| -> Result<[f32; 3]> {
                    ensure!(following.is_none(), "body following owns torso commands");
                    ensure!(
                        fields.len() == 4 && fields[0] == "TORSO",
                        "expected TORSO <roll> <pitch> <yaw> in radians"
                    );
                    let values = [fields[1].parse()?, fields[2].parse()?, fields[3].parse()?];
                    policy.validate_torso_command(values)?;
                    Ok(values)
                })();
                match requested {
                    Ok(values) => torso_command = values,
                    Err(error) => {
                        eprintln!("Torso command rejected; retaining previous target: {error}")
                    }
                }
            }
            let following_live = if let Some(following) = &mut following {
                let (live, tracked_torso) =
                    following.update(&sample.motor_position, period.as_secs_f64())?;
                sample.torso_command = policy.tracked_torso_command(tracked_torso)?;
                live
            } else {
                sample.torso_command = torso_command;
                false
            };
            let mut step = policy.step(&sample)?;
            if let Some(following) = &following {
                following.apply(&mut step.motor_target)?;
            }
            let target: Vec<f64> = step.motor_target.iter().map(|v| *v as f64).collect();
            let command = send(
                &mut controls,
                &policy,
                &target,
                &config,
                &mut targets,
                &limits,
                &mut log,
            )
            .await?;
            if let Some(log) = &mut log {
                serde_json::to_writer(
                    &mut *log,
                    &serde_json::json!({"elapsed_seconds": started.elapsed().as_secs_f64(), "feedback": sample, "policy": step, "following_live": following_live, "command": command}),
                )?;
                log.write_all(b"\n")?;
                log.flush()?;
            }
            pace(&mut deadline, period, lag).await?;
        }
        Ok::<_, anyhow::Error>(())
    };
    let result = {
        tokio::pin!(operation);
        loop {
            tokio::select! {
                result = &mut operation => break result,
                _ = interrupt.recv() => {
                    if motion_interrupt.as_ref().is_some_and(|motion| motion.request_return()) {
                        eprintln!("Motion interrupted: requesting return to standing.");
                    } else { break Err(anyhow::anyhow!("operator interrupted control")); }
                },
                _ = terminate.recv() => break Err(anyhow::anyhow!("process termination requested")),
            }
        }
    };
    if let Err(error) = &result {
        eprintln!("Control stopped: {error:#}");
    }
    let mut cleanup_errors = Vec::new();
    let stops = join_all(
        controls
            .iter_mut()
            .filter(|c| matches!(c.state(), ControlState::Running | ControlState::Uncertain))
            .map(|c| tokio::time::timeout(rpc, c.stop())),
    )
    .await;
    for stop in stops {
        match stop {
            Ok(Ok(())) => {}
            other => cleanup_errors.push(format!("stop: {other:?}")),
        }
    }
    if controls.iter().any(|c| c.state() != ControlState::Ready) {
        operator.prompt("Control is stopping; support the robot. Type IDLE to disable motors and release control:")?;
        let confirmed = tokio::select! { r = operator.confirm("IDLE") => r, _ = interrupt.recv() => Err(anyhow::anyhow!("IDLE was not confirmed")), _ = terminate.recv() => Err(anyhow::anyhow!("IDLE was not confirmed")) };
        if let Err(error) = confirmed {
            eprintln!(
                "IDLE not confirmed: no explicit disable sent; daemon timeout/lease-expiry safety remains active."
            );
            return Err(error).context(format!(
                "control result: {result:?}; cleanup: {cleanup_errors:?}"
            ));
        }
    }
    for closed in join_all(
        controls
            .into_iter()
            .map(|c| tokio::time::timeout(rpc, c.shutdown())),
    )
    .await
    {
        match closed {
            Ok(Ok(())) => {}
            other => cleanup_errors.push(format!("shutdown: {other:?}")),
        }
    }
    ensure!(
        cleanup_errors.is_empty(),
        "control result: {result:?}; cleanup failed: {cleanup_errors:?}"
    );
    result
}
