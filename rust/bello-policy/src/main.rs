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
    /// Read-only hardware preflight; does not acquire a lease or enable motors.
    #[arg(long, conflicts_with_all = ["check", "telemetry"])]
    hardware_config: Option<PathBuf>,
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
        return Ok(());
    }
    if let Some(path) = args.hardware_config {
        use bello_hardware_client::raw::proto::GroupState;
        use bello_policy::hardware::{Config, Hardware};

        let config: Config = serde_yaml::from_reader(std::fs::File::open(path)?)?;
        let mut runtime = Locomotion::new(policy)?;
        let hardware = Hardware::connect(config, &runtime).await?;
        let sample = hardware.feedback(GroupState::Idle).await?;
        let preview = runtime.step(&sample)?;
        println!(
            "{}",
            serde_json::json!({"hardware_preflight": "passed", "armed": false,
            "observation": preview.observation, "nominal_motor_target": runtime.policy.metadata.default_position})
        );
        return Ok(());
    }
    if args.telemetry {
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
