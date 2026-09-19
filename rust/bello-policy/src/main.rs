use std::io::{self, BufRead};
use std::path::PathBuf;

use anyhow::Result;
use bello_policy::Policy;
use bello_policy::locomotion::{Feedback, Locomotion};
use clap::Parser;

#[derive(Parser)]
#[command(about = "CPU ONNX policy inference; reads one stacked observation JSON array per line")]
struct Args {
    #[arg(long)]
    policy: PathBuf,
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    check: bool,
    /// Read raw hardwared feedback JSON, assembling checkpoint-defined observations.
    #[arg(long, conflicts_with = "check")]
    telemetry: bool,
    /// Hardware settings; read-only preflight unless --arm is also supplied.
    #[arg(long, conflicts_with_all = ["check", "telemetry"])]
    hardware_config: Option<PathBuf>,
    /// Requires interactive HOLD/RUN/IDLE confirmation and explicit control settings.
    #[arg(long, requires = "hardware_config")]
    arm: bool,
}

#[tokio::main(worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut policy = Policy::load(&args.policy, &args.source)?;
    eprintln!(
        "{} {}: {} motor coordinates",
        policy.metadata.robot,
        policy.metadata.git_commit,
        policy.metadata.coordinates.len()
    );
    if args.check {
        if policy.metadata.task == "motion_tracking" {
            bello_policy::motion::Motion::new(policy)?;
        }
        return Ok(());
    }
    if let Some(path) = args.hardware_config {
        use bello_hardware_client::raw::proto::GroupState;
        use bello_policy::hardware::{Config, Hardware};

        let mut config: Config = serde_yaml::from_reader(std::fs::File::open(path)?)?;
        let control = config.control.take();
        let (mut runtime, mut live) = if policy.metadata.task == "motion_tracking" {
            let settings = config.motion.take().ok_or_else(|| {
                anyhow::anyhow!("motion hardware control requires motion settings")
            })?;
            let standing = Locomotion::new(Policy::load(&settings.standing_policy, &args.source)?)?;
            let live = bello_policy::motion::live::Live::connect(
                bello_policy::motion::Motion::new(policy)?,
                &standing,
                settings,
            )
            .await?;
            (standing, Some(live))
        } else {
            anyhow::ensure!(
                config.motion.is_none(),
                "motion settings require a motion bundle"
            );
            (Locomotion::new(policy)?, None)
        };
        let hardware = Hardware::connect(config, &runtime).await?;
        if let Some(live) = &mut live {
            live.configure_target_limits(
                hardware.target_limits.clone(),
                hardware.restrict_target_limits(live.motion_limits())?,
            )?;
            anyhow::ensure!(
                hardware
                    .groups
                    .iter()
                    .all(|group| group.control_rate_hz() + 1e-6 >= 1. / live.motion_period()),
                "hardware control rate is slower than the motion policy"
            );
        }
        if args.arm {
            return bello_policy::control::run(
                hardware,
                runtime,
                control
                    .ok_or_else(|| anyhow::anyhow!("armed execution requires control settings"))?,
                live,
            )
            .await;
        }
        let sample = hardware.feedback(GroupState::Idle).await?;
        let startup_target = hardware.target_limits.startup_target(
            &sample.motor_position,
            hardware.config.startup_feedback_margin_radians,
        )?;
        if let Some(live) = &live {
            live.check_feedback(&sample)?;
        }
        let preview = runtime.step(&sample)?;
        println!(
            "{}",
            serde_json::json!({"hardware_preflight": "passed", "armed": false,
            "observation": preview.observation, "nominal_motor_target": runtime.policy.metadata.default_position,
            "startup_motor_target": startup_target})
        );
        return Ok(());
    }
    if args.telemetry {
        if policy.metadata.task == "motion_tracking" {
            let mut runtime = bello_policy::motion::Motion::new(policy)?;
            for line in io::stdin().lock().lines() {
                let sample = serde_json::from_str(&line?)?;
                println!("{}", serde_json::to_string(&runtime.step(&sample)?)?);
            }
            return Ok(());
        }
        let mut runtime = Locomotion::new(policy)?;
        for line in io::stdin().lock().lines() {
            let sample: Feedback = serde_json::from_str(&line?)?;
            println!("{}", serde_json::to_string(&runtime.step(&sample)?)?);
        }
        return Ok(());
    }
    for line in io::stdin().lock().lines() {
        let observation: Vec<f32> = serde_json::from_str(&line?)?;
        println!("{}", serde_json::to_string(&policy.infer(&observation)?)?);
    }
    Ok(())
}
