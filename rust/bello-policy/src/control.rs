//! Explicit operator lifecycle over bello-stack's lease-owning control sessions.
use crate::{
    hardware::Hardware,
    locomotion::Locomotion,
    operator::{self, RequestMode, proto::Mode},
};
use anyhow::{Context, Result, ensure};
use bello_hardware_client::{
    CommandFrame, ControlOptions, ControlSession, CoordinateTarget, MotorGroup, raw::proto,
};
use futures_util::future::join_all;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub following: crate::following::Config,
    pub client_name: String,
    pub operator_listen: std::net::SocketAddr,
    pub command_validity_seconds: f64,
    pub acknowledgement_timeout_seconds: f64,
    pub rpc_timeout_seconds: f64,
    pub ramp_seconds: f64,
    pub gain_scale: f64,
    pub joystick_command_scale: f32,
    #[serde(default)]
    pub torso_command: [f32; 3],
    pub max_loop_lag_seconds: f64,
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
        Ok(())
    }
}

fn frame(
    group: &MotorGroup,
    coordinates: &[String],
    target: &[f64],
    gains: &[[f32; 2]],
    scale: f64,
) -> Result<CommandFrame> {
    ensure!(
        target.len() == coordinates.len() && gains.len() == target.len(),
        "motor target/gain shape mismatch"
    );
    let mut builder = group.frame();
    for motor in group.motors() {
        let i = coordinates
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
                .gains(gains[i][0] as f64 * scale, gains[i][1] as f64 * scale),
        )?;
    }
    Ok(builder.build()?)
}

async fn send(
    controls: &mut [ControlSession],
    target: &[f64],
    gains: &[[f32; 2]],
    config: &Config,
    targets: &mut CommandTargets,
    limits: &crate::target_limits::TargetLimits,
    log: &mut Option<File>,
) -> Result<crate::target_limits::LimitedTarget> {
    let command = limits.constrain(target, &targets.previous)?;
    let frames = controls
        .iter()
        .map(|c| {
            frame(
                c.group(),
                &targets.coordinates,
                &command.motor_target,
                gains,
                config.gain_scale,
            )
        })
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
    coordinates: Vec<String>,
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

async fn recover_groups(hardware: &Hardware, config: &Config) -> Result<()> {
    let rpc = Duration::from_secs_f64(config.rpc_timeout_seconds);
    for group in &hardware.groups {
        let mut control = tokio::time::timeout(rpc, group.control(&config.client_name))
            .await
            .context("recovery lease acquisition timed out")??;
        let recovery = async {
            let inventory = hardware.robot.raw_client().inventory().await?;
            let current = inventory
                .groups
                .iter()
                .find(|item| item.name == group.name())
                .context("recovery group missing from hardware inventory")?;
            if current.state == proto::GroupState::Fault as i32 {
                control.recover().await?;
            } else {
                control.disable().await?;
            }
            Ok::<_, anyhow::Error>(())
        };
        let result = tokio::time::timeout(rpc, recovery)
            .await
            .context("group recovery timed out")
            .and_then(|value| value);
        let cleanup = tokio::time::timeout(rpc, control.release())
            .await
            .context("recovery lease release timed out")
            .and_then(|value| value.map_err(Into::into));
        result.with_context(|| format!("{} recovery failed", group.name()))?;
        cleanup?;
    }
    hardware.feedback(proto::GroupState::Idle).await?;
    Ok(())
}

pub async fn run(
    hardware: Hardware,
    mut policy: Locomotion,
    config: Config,
    mut motions: BTreeMap<String, crate::motion::live::Live>,
) -> Result<()> {
    config.validate()?;
    policy.controller_command([0.; 3], config.joystick_command_scale)?;
    ensure!(
        config.command_validity_seconds > policy.policy.metadata.control_timestep,
        "command validity must exceed the policy period"
    );
    ensure!(
        config.torso_command == [0.; 3],
        "operator control requires a neutral standing torso command"
    );
    let standing_gains: Vec<[f32; 2]> = policy
        .policy
        .metadata
        .kp
        .iter()
        .zip(&policy.policy.metadata.kd)
        .map(|(&kp, &kd)| [kp, kd])
        .collect();
    let rpc = Duration::from_secs_f64(config.rpc_timeout_seconds);
    let operator =
        operator::Source::bind(config.operator_listen, motions.keys().cloned().collect())?;
    operator.set_tracking(config.following.enabled, false, false);
    let mut log: Option<File> = config
        .policy_log_path
        .as_ref()
        .map(|path| OpenOptions::new().write(true).create_new(true).open(path))
        .transpose()?;
    let power = log
        .as_ref()
        .map(|_| crate::power::Source::start(&hardware.robot, rpc));
    let preflight = hardware.ready_feedback().await?;
    let limits = hardware.target_limits.clone();
    let startup_margin = hardware.config.startup_feedback_margin_radians;
    limits.startup_target(&preflight.motor_position, startup_margin)?;
    for motion in motions.values() {
        motion.check_feedback(&preflight)?;
    }
    policy.step(&preflight)?;
    policy.reset();
    eprintln!(
        "Operator service listening on {}; waiting for neutral input and Back+Start.",
        config.operator_listen
    );
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    for motion in motions.values() {
        ensure!(
            config.command_validity_seconds > motion.motion_period(),
            "command validity must exceed every motion policy period"
        );
    }
    // Each recovery starts a fresh control session, while the operator service
    // and policy log remain open. No lease is acquired until another entry chord.
    'sessions: loop {
        let mut following = if config.following.enabled {
            Some(crate::following::Following::new(
                config.following.clone(),
                &policy.policy,
            )?)
        } else {
            None
        };
        let mut controls = Vec::new();
        let motion_interrupt = Arc::new(Mutex::new(None::<crate::motion::live::Interrupt>));
        let operation = async {
            let (startup_target, measured_startup) = loop {
                loop {
                    let input = operator.snapshot();
                    ensure!(input.request != Some(RequestMode::Abort), "operator abort");
                    if !input.lost && input.request == Some(RequestMode::Hold) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                let initial = hardware.ready_feedback().await?;
                let startup_target =
                    limits.startup_target(&initial.motor_position, startup_margin)?;
                for group in &hardware.groups {
                    let input = operator.snapshot();
                    ensure!(input.request != Some(RequestMode::Abort), "operator abort");
                    if input.lost {
                        break;
                    }
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
                let input = operator.snapshot();
                ensure!(input.request != Some(RequestMode::Abort), "operator abort");
                if input.lost || controls.len() != hardware.groups.len() {
                    for control in controls.drain(..) {
                        tokio::time::timeout(rpc, control.release())
                            .await
                            .context("lease release timed out")??;
                    }
                    operator.set_mode(Mode::Disarmed, "disarmed", "", 0.);
                    continue;
                }
                break (startup_target, initial.motor_position);
            };
            if let Some(log) = &mut log {
                serde_json::to_writer(
                    &mut *log,
                    &serde_json::json!({"event": "startup_target",
                "motor_target": startup_target, "measured_motor_position": measured_startup,
                "motor_rails": power.as_ref().map(crate::power::Source::snapshot)}),
                )?;
                log.write_all(b"\n")?;
                log.flush()?;
            }
            let mut targets = CommandTargets {
                coordinates: policy.policy.metadata.coordinates.clone(),
                previous: startup_target.clone(),
                limited_count: 0,
            };
            let frames = controls
                .iter()
                .map(|control| {
                    frame(
                        control.group(),
                        &targets.coordinates,
                        &startup_target,
                        &standing_gains,
                        config.gain_scale,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            for result in join_all(
                controls
                    .iter_mut()
                    .zip(frames)
                    .map(|(control, frame)| tokio::time::timeout(rpc, control.start(frame))),
            )
            .await
            {
                result.context("motor start timed out")??;
            }
            operator.set_mode(Mode::Hold, "ramping", "", 0.);
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
            let mut step = 0;
            while step < steps {
                let input = operator.snapshot();
                ensure!(input.request != Some(RequestMode::Abort), "operator abort");
                hardware.feedback(proto::GroupState::Running).await?;
                let ramp = if input.lost {
                    targets.previous.clone()
                } else {
                    step += 1;
                    startup_target
                        .iter()
                        .zip(&target)
                        .map(|(a, b)| a + step as f64 / steps as f64 * (b - a))
                        .collect()
                };
                send(
                    &mut controls,
                    &ramp,
                    &standing_gains,
                    &config,
                    &mut targets,
                    &limits,
                    &mut log,
                )
                .await?;
                operator.set_mode(Mode::Hold, "ramping", "", step as f64 / steps as f64);
                pace(&mut deadline, period, lag).await?;
            }
            operator.set_mode(Mode::Hold, "holding", "", 1.);
            loop {
                let input = operator.snapshot();
                ensure!(input.request != Some(RequestMode::Abort), "operator abort");
                hardware.feedback(proto::GroupState::Running).await?;
                let held = if input.lost {
                    targets.previous.clone()
                } else {
                    target.clone()
                };
                send(
                    &mut controls,
                    &held,
                    &standing_gains,
                    &config,
                    &mut targets,
                    &limits,
                    &mut log,
                )
                .await?;
                if input.request == Some(RequestMode::Stand) {
                    break;
                }
                pace(&mut deadline, period, lag).await?;
            }
            policy.reset();
            operator.set_mode(Mode::Stand, "standing", "", 0.);
            let started = Instant::now();
            deadline = tokio::time::Instant::now();
            let mut active_period = period;
            let mut mode = Mode::Stand;
            let mut active_motion: Option<String> = None;
            let mut tracking_enabled = false;
            operator.set_tracking(following.is_some(), false, false);
            loop {
                let input = operator.snapshot();
                ensure!(input.request != Some(RequestMode::Abort), "operator abort");
                if input.lost && mode == Mode::Locomotion {
                    mode = Mode::Stand;
                    policy.reset();
                    operator.set_mode(mode, "standing", "", 0.);
                }
                let mut start_motion = false;
                match input.request {
                    Some(RequestMode::SetTracking(enabled))
                        if mode == Mode::Locomotion && following.is_some() =>
                    {
                        tracking_enabled = enabled;
                    }
                    Some(RequestMode::Stand) if mode == Mode::Locomotion => {
                        mode = Mode::Stand;
                        policy.reset();
                        operator.set_mode(mode, "standing", "", 0.);
                    }
                    Some(RequestMode::Locomotion) if mode == Mode::Stand => {
                        mode = Mode::Locomotion;
                        policy.reset();
                        operator.set_mode(mode, "locomotion", "", 0.);
                    }
                    Some(RequestMode::Motion(name))
                        if mode == Mode::Stand && motions.contains_key(&name) =>
                    {
                        *motion_interrupt.lock().unwrap() = Some(motions[&name].interrupt.clone());
                        active_motion = Some(name);
                        start_motion = true;
                        mode = Mode::Motion;
                        if let Some(following) = &mut following {
                            following.reset();
                        }
                    }
                    _ => {}
                }
                if mode != Mode::Locomotion {
                    tracking_enabled = false;
                    operator.set_tracking(following.is_some(), false, false);
                }
                let max_tilt = active_motion
                    .as_ref()
                    .and_then(|name| motions.get(name))
                    .map_or(Some(hardware.config.max_tilt_radians), |motion| {
                        motion.absolute_tilt_limit(hardware.config.max_tilt_radians)
                    });
                let mut sample = hardware
                    .feedback_with_tilt(proto::GroupState::Running, max_tilt)
                    .await?;
                if let Some(name) = active_motion.as_ref() {
                    let motion = motions
                        .get_mut(name)
                        .expect("selected motion was preloaded");
                    let step = motion.step(
                        &mut policy,
                        &sample,
                        start_motion,
                        &targets.previous,
                        hardware.config.max_tilt_radians,
                    )?;
                    if step.period != active_period {
                        deadline = tokio::time::Instant::now();
                        active_period = step.period;
                    }
                    let phase = if let Some(failure) = step.log["failure"].as_str() {
                        format!("returning to Stand: {failure}")
                    } else {
                        step.log["phase"].as_str().unwrap_or("motion").to_owned()
                    };
                    let progress = step.log["reference_time"]
                        .as_f64()
                        .map_or(0., |time| time / motion.duration());
                    let target: Vec<f64> = step.target.iter().map(|value| *value as f64).collect();
                    let command = send(
                        &mut controls,
                        &target,
                        &step.motor_gains,
                        &config,
                        &mut targets,
                        motion.target_limits(),
                        &mut log,
                    )
                    .await?;
                    if let Some(log) = &mut log {
                        serde_json::to_writer(
                            &mut *log,
                            &serde_json::json!({"elapsed_seconds": started.elapsed().as_secs_f64(),
                        "feedback": sample, "motion": step.log, "command": command,
                        "motor_rails": power.as_ref().map(crate::power::Source::snapshot)}),
                        )?;
                        log.write_all(b"\n")?;
                        log.flush()?;
                    }
                    if motion.is_standing() {
                        *motion_interrupt.lock().unwrap() = None;
                        active_motion = None;
                        mode = Mode::Stand;
                        policy.reset();
                        operator.set_mode(mode, "standing", "", 0.);
                    } else {
                        operator.set_mode(mode, &phase, name, progress);
                    }
                    pace(&mut deadline, active_period, lag).await?;
                    continue;
                }
                if active_period != period {
                    active_period = period;
                    deadline = tokio::time::Instant::now();
                }
                sample.command = policy.controller_command(
                    if mode == Mode::Locomotion {
                        input.axes
                    } else {
                        [0.; 3]
                    },
                    config.joystick_command_scale,
                )?;
                let following_live = if let Some(following) = &mut following {
                    let (live, tracked_torso) = following.update(
                        &sample.motor_position,
                        period.as_secs_f64(),
                        tracking_enabled,
                    )?;
                    sample.torso_command = policy.tracked_torso_command(tracked_torso)?;
                    live
                } else {
                    false
                };
                operator.set_tracking(following.is_some(), tracking_enabled, following_live);
                let mut step = policy.step(&sample)?;
                if let Some(following) = &following {
                    following.apply(&mut step.motor_target)?;
                }
                let target: Vec<f64> = step
                    .motor_target
                    .iter()
                    .map(|value| *value as f64)
                    .collect();
                let command = send(
                    &mut controls,
                    &target,
                    &standing_gains,
                    &config,
                    &mut targets,
                    &limits,
                    &mut log,
                )
                .await?;
                operator.set_velocity_command(policy.operator_velocity(sample.command));
                if let Some(log) = &mut log {
                    serde_json::to_writer(
                        &mut *log,
                        &serde_json::json!({"elapsed_seconds": started.elapsed().as_secs_f64(),
                    "feedback": sample, "policy": step, "following_live": following_live,
                    "command": command, "motor_rails": power.as_ref().map(crate::power::Source::snapshot)}),
                    )?;
                    log.write_all(b"\n")?;
                    log.flush()?;
                }
                pace(&mut deadline, period, lag).await?;
            }
        };
        let (result, shutdown_requested): (Result<()>, bool) = {
            tokio::pin!(operation);
            loop {
                tokio::select! {
                    result = &mut operation => break (result, false),
                    _ = interrupt.recv() => {
                        if motion_interrupt.lock().unwrap().as_ref().is_some_and(|motion| motion.request_return()) {
                            eprintln!("Motion interrupted: requesting return to standing.");
                        } else {
                            break (Err(anyhow::anyhow!("operator interrupted control")), true);
                        }
                    },
                    _ = terminate.recv() => break (Err(anyhow::anyhow!("process termination requested")), true),
                }
            }
        };
        if let Err(error) = &result {
            operator.set_tracking(config.following.enabled, false, false);
            eprintln!("Control stopped: {error:#}");
            operator.fault(&format!("{error:#}"));
        }
        let mut cleanup_errors = Vec::new();
        for closed in join_all(
            controls
                .into_iter()
                .map(|control| tokio::time::timeout(rpc, control.release())),
        )
        .await
        {
            if let Err(error) = closed
                .context("damping release timed out")
                .and_then(|value| value.map_err(Into::into))
            {
                cleanup_errors.push(format!("damping release: {error:#}"));
            }
        }
        if !cleanup_errors.is_empty() {
            operator.fault(&format!(
                "control result: {result:?}; cleanup failed: {cleanup_errors:?}"
            ));
        }
        if !shutdown_requested {
            operator.allow_fault_clear("");
            loop {
                tokio::select! {
                    _ = interrupt.recv() => break,
                    _ = terminate.recv() => break,
                    _ = tokio::time::sleep(Duration::from_millis(20)) => {}
                }
                if !operator.take_fault_clear() {
                    continue;
                }
                match recover_groups(&hardware, &config).await {
                    Ok(()) => {
                        if !operator.disarm_after_recovery() {
                            operator.allow_fault_clear("Operator abort during fault recovery.");
                            continue;
                        }
                        policy.reset();
                        policy.controller_command([0.; 3], config.joystick_command_scale)?;
                        for motion in motions.values_mut() {
                            motion.reset();
                        }
                        if let Some(log) = &mut log {
                            writeln!(log, "{{\"event\":\"fault_cleared\"}}")?;
                            log.flush()?;
                        }
                        continue 'sessions;
                    }
                    Err(error) => operator.allow_fault_clear(&format!("{error:#}")),
                }
            }
        }
        ensure!(
            cleanup_errors.is_empty(),
            "control result: {result:?}; cleanup failed: {cleanup_errors:?}"
        );
        return result;
    }
}
