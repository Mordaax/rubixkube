//! Product commands; all cluster operations require an explicit connection.
use crate::{
    cluster::{Api, Cluster},
    config::{self, ExperimentConfig},
    experiment, invalid, metrics, Result,
};
use clap::{Args, Subcommand};
use serde_json::{json, Value};
use std::{fs, io::Write, path::PathBuf, time::Duration};
#[derive(Debug, Clone, Args)]
pub struct Connection {
    /// Explicit kubeconfig; never falls back to an implicit cluster.
    #[arg(long)]
    pub kubeconfig: PathBuf,
    /// Exact kubeconfig context to use.
    #[arg(long)]
    pub context: String,
    /// Defaults to config namespace, or default for commands without a config.
    #[arg(long)]
    pub namespace: Option<String>,
}
impl Connection {
    fn cluster(&self, config: Option<&ExperimentConfig>) -> Result<Cluster> {
        if let (Some(ns), Some(c)) = (&self.namespace, config) {
            if ns != &c.namespace {
                return Err(invalid("--namespace differs from configuration namespace"));
            }
        }
        Cluster::new(
            self.kubeconfig.clone(),
            self.context.clone(),
            self.namespace.clone().unwrap_or_else(|| {
                config
                    .map(|c| c.namespace.clone())
                    .unwrap_or_else(|| "default".into())
            }),
        )
    }
}
#[derive(Debug, Args)]
pub struct Named {
    pub name: String,
    #[command(flatten)]
    pub connection: Connection,
}
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Display application and bundled-font license notices.
    Licenses,
    /// Create an editable experiment configuration; refuses to overwrite a file.
    Init {
        #[arg(long, default_value = "rubixkube.json")]
        file: PathBuf,
        #[arg(long)]
        gpu: bool,
    },
    /// Validate configuration locally, optionally against API-server admission.
    Validate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        server: bool,
        #[arg(long)]
        kubeconfig: Option<PathBuf>,
        #[arg(long)]
        context: Option<String>,
    },
    /// Submit an arbitrary training image and search; optional wait is resumable.
    Submit {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        output: Option<PathBuf>,
        #[command(flatten)]
        connection: Connection,
    },
    /// List only RubixKube-managed OptimizationJobs in this namespace.
    List {
        #[command(flatten)]
        connection: Connection,
    },
    /// Inspect progress, conditions and the upstream selected result.
    Status(Named),
    /// Resume watching an existing job without creating more trials.
    Watch {
        #[command(flatten)]
        job: Named,
        #[arg(long, default_value_t = 600)]
        timeout: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Verify completed results and export JSON, Pod evidence, CSV and static plots.
    Results {
        #[command(flatten)]
        job: Named,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Retrieve worker logs for all observed trials as a JSON map.
    Logs(Named),
    /// Cancel a managed job; shared PVC data is retained unless explicitly purged.
    #[command(alias = "delete")]
    Cancel {
        #[command(flatten)]
        job: Named,
        #[arg(long)]
        purge_shared_data: bool,
    },
    /// Check API installation, create permission, runtime and advertised GPU capacity.
    Doctor {
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(flatten)]
        connection: Connection,
    },
    /// Push a finite objective from any training container; requires Trainer env.
    Report {
        #[arg(long)]
        metric: String,
        #[arg(long, allow_hyphen_values = true)]
        value: f64,
    },
}
fn print(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
fn output(name: &str, path: Option<PathBuf>) -> PathBuf {
    path.unwrap_or_else(|| PathBuf::from("runs").join(name))
}
pub fn execute(command: Commands) -> Result<()> {
    match command {
        Commands::Licenses => {
            println!(
                "{}\n{}",
                include_str!("../LICENSE"),
                include_str!("../assets/FONT-LICENSE.txt")
            );
        }
        Commands::Init { file, gpu } => {
            let data = serde_json::to_vec_pretty(&config::example(gpu))?;
            let mut handle = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&file)?;
            handle.write_all(&data)?;
            handle.write_all(b"\n")?;
            eprintln!(
                "Wrote {}. Set the runtime, image, objective and parameters, then validate.",
                file.display()
            );
        }
        Commands::Validate {
            config,
            server,
            kubeconfig,
            context,
        } => {
            let c = ExperimentConfig::read(&config)?;
            let manifest = config::manifest(&c, "validation")?;
            if server {
                let kubeconfig = kubeconfig
                    .ok_or_else(|| invalid("--server requires --kubeconfig and --context"))?;
                let context = context
                    .ok_or_else(|| invalid("--server requires --kubeconfig and --context"))?;
                let cluster = Cluster::new(kubeconfig, context, c.namespace.clone())?;
                cluster.execute(
                    &["create", "--dry-run=server", "-f", "-", "-o", "json"],
                    Some(&manifest),
                    Duration::from_secs(30),
                )?;
            }
            print(&json!({"valid":true,"server_validated":server,"config":config}))?;
        }
        Commands::Submit {
            config,
            name,
            wait,
            dry_run,
            output: directory,
            connection,
        } => {
            let c = ExperimentConfig::read(&config)?;
            let name =
                name.unwrap_or_else(|| format!("rk-{}", chrono::Utc::now().timestamp_millis()));
            let manifest = config::manifest(&c, &name)?;
            if dry_run {
                print(&manifest)?;
                return Ok(());
            }
            let api = connection.cluster(Some(&c))?;
            let dir = output(&name, directory);
            let created = experiment::submit(&api, &c, &name, &dir)?;
            if wait {
                let report =
                    experiment::watch(&api, &name, Duration::from_secs(c.timeout_seconds), &dir)?;
                print(&report)?;
            } else {
                print(
                    &json!({"name":name,"namespace":c.namespace,"uid":created["metadata"]["uid"],"output":dir,"resume":"Use watch with this name, kubeconfig, context and namespace"}),
                )?;
            }
        }
        Commands::List { connection } => print(&experiment::list(&connection.cluster(None)?)?)?,
        Commands::Status(job) => print(&experiment::status(
            &job.connection.cluster(None)?,
            &job.name,
        )?)?,
        Commands::Watch {
            job,
            timeout,
            output: directory,
        } => {
            if timeout == 0 || timeout > 86400 {
                return Err(invalid("Watch timeout must be 1..86400 seconds"));
            }
            let dir = output(&job.name, directory);
            print(&experiment::watch(
                &job.connection.cluster(None)?,
                &job.name,
                Duration::from_secs(timeout),
                &dir,
            )?)?;
        }
        Commands::Results {
            job,
            output: directory,
        } => {
            let dir = output(&job.name, directory);
            print(&experiment::collect(
                &job.connection.cluster(None)?,
                &job.name,
                &dir,
            )?)?;
        }
        Commands::Logs(job) => print(&experiment::logs(
            &job.connection.cluster(None)?,
            &job.name,
        )?)?,
        Commands::Cancel {
            job,
            purge_shared_data,
        } => print(&experiment::cancel(
            &job.connection.cluster(None)?,
            &job.name,
            purge_shared_data,
        )?)?,
        Commands::Doctor { config, connection } => {
            let c = config
                .as_ref()
                .map(|p| ExperimentConfig::read(p))
                .transpose()?;
            let report = experiment::doctor(&connection.cluster(c.as_ref())?, c.as_ref())?;
            print(&report)?;
            if report["ready"] != true {
                return Err(invalid(
                    "One or more preflight checks failed; inspect the JSON checks above",
                ));
            }
        }
        Commands::Report { metric, value } => {
            let attempt =
                metrics::deliver_metric(&metrics::MetricsConfig::from_env()?, &metric, value)?;
            print(
                &json!({"event":"metric_delivered","metric":metric,"value":value,"attempt":attempt}),
            )?;
        }
    }
    Ok(())
}
