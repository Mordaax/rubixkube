//! Command-line entry point for the Rust OptimizationJob demonstration.
use clap::{Parser, Subcommand};
use optimizationjob::{
    invalid,
    metrics::{deliver, MetricsConfig},
    orchestrate::{self, RunOptions},
    serve,
    training::train,
    Result,
};
use serde_json::json;
use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Parser)]
#[command(
    version,
    about = "Real Kubernetes HPO demo: Rust training, metrics and evidence tooling"
)]
struct Cli {
    /// Repository root (discovered from current directory or executable when omitted).
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Fit the fixed dataset; report final loss to Trainer unless --local is set.
    Train {
        #[arg(long)]
        learning_rate: Option<f64>,
        #[arg(long, default_value_t = 40)]
        epochs: usize,
        /// Train locally without claiming Kubernetes metric delivery.
        #[arg(long)]
        local: bool,
        /// Execution device; CUDA errors never fall back to CPU.
        #[arg(long, default_value = "cpu", value_parser = ["cpu","cuda"])]
        device: String,
    },
    /// Submit/watch a real job and save independently verified report artifacts.
    Run(RunArgs),
    /// Build images, install pinned upstream controllers and run the Rust demo (Linux/WSL).
    Start(RunArgs),
    /// Verify clean pinned Trainer source and run its real Go baseline tests.
    Smoke {
        #[arg(long)]
        trainer: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Serve saved results on 127.0.0.1 with path containment.
    Serve {
        #[arg(long)]
        directory: Option<PathBuf>,
        #[arg(long, default_value_t = 8765)]
        port: u16,
    },
}
#[derive(clap::Args)]
struct RunArgs {
    /// Use the dedicated GPU cluster/runtime and require CUDA evidence.
    #[arg(long)]
    gpu: bool,
    /// Inspect a previously completed Rust job instead of submitting another.
    #[arg(long)]
    existing: Option<String>,
    #[arg(long)]
    kubeconfig: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 300)]
    timeout: u64,
}
fn root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        let path = path.canonicalize()?;
        if path.join("upstream.lock.json").is_file() {
            return Ok(path);
        }
        return Err(invalid(
            "--root must point to the optimizationjob repository",
        ));
    }
    for path in [env::current_dir()?, env::current_exe()?] {
        if let Some(found) = path
            .ancestors()
            .find(|p| p.join("upstream.lock.json").is_file())
        {
            return Ok(found.to_path_buf());
        }
    }
    Err(invalid("Cannot locate repository; supply --root"))
}
fn options(root: &Path, args: RunArgs) -> Result<RunOptions> {
    Ok(RunOptions {
        existing: args.existing,
        gpu: args.gpu,
        kubeconfig: args.kubeconfig.unwrap_or_else(|| {
            root.join(if args.gpu {
                ".tools/gpu/kubeconfig"
            } else {
                ".tools/demo/kubeconfig"
            })
        }),
        output: args.output.unwrap_or(if args.gpu {
            root.parent()
                .ok_or_else(|| invalid("No workspace parent"))?
                .join("optimizationjob-evidence/runs/gpu/reports")
        } else {
            orchestrate::report_directory(root)?
        }),
        timeout: Duration::from_secs(args.timeout),
    })
}
fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Action::Train {
            learning_rate,
            epochs,
            local,
            device,
        } => {
            let rate = match learning_rate {
                Some(rate) => rate,
                None => env::var("KUBEFLOW_TRAINER_OPT_learning_rate")
                    .map_err(|_| {
                        invalid("Specify --learning-rate or run inside a configured TrainJob")
                    })?
                    .parse()
                    .map_err(|_| invalid("Learning rate must be numeric"))?,
            };
            if !local && epochs != 40 {
                return Err(invalid(
                    "Kubernetes demo uses exactly 40 epochs; custom epochs require --local",
                ));
            }
            let result = if device == "cuda" {
                #[cfg(feature = "cuda")]
                {
                    optimizationjob::gpu::train_cuda(rate, epochs)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(invalid("Rebuild with --features cuda to use GPU training"));
                }
            } else {
                train(rate, epochs)?
            };
            let mut event = serde_json::to_value(&result)?;
            event["event"] = json!("training_complete");
            println!("{event}");
            io::stdout().flush()?;
            if !local {
                let attempt = deliver(&MetricsConfig::from_env()?, result.val_loss)?;
                println!("{}", json!({"event":"metric_delivered","attempt":attempt}));
                io::stdout().flush()?;
            }
        }
        Action::Run(args) => {
            let root = root(cli.root)?;
            orchestrate::run(&root, &options(&root, args)?)?;
        }
        Action::Start(args) => {
            let root = root(cli.root)?;
            orchestrate::start(&root, &options(&root, args)?)?;
        }
        Action::Smoke { trainer, output } => {
            let root = root(cli.root)?;
            let output = output.unwrap_or(
                root.parent()
                    .ok_or_else(|| invalid("No workspace parent"))?
                    .join("optimizationjob-evidence/runs/rust/baseline"),
            );
            orchestrate::smoke(&root, trainer.as_deref(), &output)?;
        }
        Action::Serve { directory, port } => {
            let directory = match directory {
                Some(path) => path,
                None => orchestrate::report_directory(&root(cli.root)?)?,
            };
            serve::serve(&directory, port)?;
        }
    }
    Ok(())
}
fn main() {
    if let Err(error) = execute(Cli::parse()) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
