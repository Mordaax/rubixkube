// Command-line entry point for RubixKube and its compatibility binary.
use clap::{Parser, Subcommand};
use rubixkube::{
    invalid,
    metrics::{deliver_metric, MetricsConfig},
    orchestrate::{self, RunOptions},
    plots, Result,
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
    name = "rubixkube",
    about = "Configurable Kubernetes hyperparameter optimization with CPU and CUDA workloads"
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
    #[command(flatten)]
    Product(rubixkube::cli_product::Commands),
    /// Generate shell completions for installation.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    /// Fit the fixed dataset; report final loss to Trainer unless --local is set.
    Train {
        #[arg(long)]
        learning_rate: Option<f64>,
        #[arg(long)]
        epochs: Option<usize>,
        /// Train locally without claiming Kubernetes metric delivery.
        #[arg(long)]
        local: bool,
        /// Execution device; CUDA errors never fall back to CPU.
        #[arg(long, default_value = "cpu", value_parser = ["cpu","cuda"])]
        device: String,
        /// CSV with x,y columns (first half training, second half validation).
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Metric name reported to Trainer.
        #[arg(long, default_value = "val_loss")]
        metric: String,
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
    /// Export simple PNG/SVG plots and CSV from a saved verified report.
    Plot {
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
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
        Action::Product(command) => rubixkube::cli_product::execute(command)?,
        Action::Completions { shell } => {
            use clap::CommandFactory;
            clap_complete::generate(shell, &mut Cli::command(), "rubixkube", &mut io::stdout());
        }
        Action::Train {
            learning_rate,
            epochs,
            local,
            device,
            dataset,
            metric,
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
            let epochs = match epochs {
                Some(n) => n,
                None => env::var("KUBEFLOW_TRAINER_OPT_epochs")
                    .ok()
                    .map(|s| s.parse::<usize>())
                    .transpose()
                    .map_err(|_| invalid("epochs parameter must be a positive integer"))?
                    .unwrap_or(40),
            };
            let samples = match &dataset {
                Some(path) => rubixkube::dataset::read_csv(path)?,
                None => rubixkube::dataset::embedded()?,
            };
            let result = if device == "cuda" {
                #[cfg(feature = "cuda")]
                {
                    rubixkube::gpu::train_cuda_on(rate, epochs, &samples)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(invalid("Rebuild with --features cuda to use GPU training"));
                }
            } else {
                rubixkube::training::train_on(rate, epochs, &samples)?
            };
            let mut event = serde_json::to_value(&result)?;
            event["event"] = json!("training_complete");
            event["objective_metric"] = json!(metric);
            event["dataset_rows"] = json!(samples.len());
            event["dataset_source"] = json!(dataset
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "embedded".into()));
            println!("{event}");
            io::stdout().flush()?;
            if !local {
                let attempt =
                    deliver_metric(&MetricsConfig::from_env()?, &metric, result.val_loss)?;
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
        Action::Plot { report, output } => {
            let data = serde_json::from_slice(&std::fs::read(&report)?)?;
            let directory =
                output.unwrap_or_else(|| report.parent().unwrap_or(Path::new(".")).to_path_buf());
            plots::write_plots(&data, &directory)?;
            println!("Saved plots and results to {}", directory.display());
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
