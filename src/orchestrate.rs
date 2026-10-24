//! Submit real TrainJobs, collect checked evidence, and reproduce the local cluster.
use crate::{
    invalid,
    process::{capture, checked, verify_checkout, wait_for},
    report::{condition, string, summarize},
    training::TrainingResult,
    Result,
};
use serde_json::{json, Value};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

const CONTEXT: &str = "kind-optimizationjob-demo";
const NAMESPACE: &str = "optimizationjob-demo";
const TRAINING_IMAGE: &str = "optimizationjob-demo-training:rust";
const PROVIDER: &str = "ghcr.io/kubeflow/katib/suggestion-optuna@sha256:04a71f288d04251835846b87d18a9fcf816eb59003d282555dc3d94412c879c7";
const PROVIDER_TAG: &str = "ghcr.io/kubeflow/katib/suggestion-optuna:v0.19.0";

/// Options for submitting or inspecting one real demo run.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub gpu: bool,
    pub existing: Option<String>,
    pub kubeconfig: PathBuf,
    pub output: PathBuf,
    pub timeout: Duration,
}
/// Default evidence directory stays outside the Git repository and old reports.
pub fn report_directory(root: &Path) -> Result<PathBuf> {
    Ok(root
        .parent()
        .ok_or_else(|| invalid("Repository must have a parent"))?
        .join("optimizationjob-evidence/runs/rust/reports"))
}
pub(crate) fn kube(
    config: &Path,
    gpu: bool,
    args: &[&str],
    payload: Option<&Value>,
) -> Result<String> {
    let bytes = payload.map(serde_json::to_vec).transpose()?;
    checked(
        Command::new("kubectl")
            .arg("--kubeconfig")
            .arg(config)
            .args(["--context", context(gpu)])
            .args(args),
        bytes.as_deref(),
        Duration::from_secs(45),
    )
}
fn kube_json(config: &Path, gpu: bool, args: &[&str]) -> Result<Value> {
    Ok(serde_json::from_str(&kube(config, gpu, args, None)?)?)
}
fn array(value: &Value) -> Result<&Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid("Expected resource list"))
}
fn dns_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(|part| {
            !part.is_empty()
                && part.len() <= 63
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                && part.as_bytes()[0] != b'-'
                && part.as_bytes()[part.len() - 1] != b'-'
        })
}
fn save(directory: &Path, name: &str, contents: impl AsRef<[u8]>) -> Result<()> {
    let path = directory.join(name);
    if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(invalid(
            "Refusing to overwrite symlink in evidence directory",
        ));
    }
    fs::write(path, contents)?;
    Ok(())
}
fn save_json(directory: &Path, name: &str, data: &Value) -> Result<()> {
    save(directory, name, serde_json::to_vec_pretty(data)?)
}

/// Require a Rust worker's measured training and explicit acknowledgement events.
pub fn validate_worker(row: &Value, logs: &str) -> Result<TrainingResult> {
    validate_worker_mode(row, logs, false)
}
/// Verify execution-mode-specific evidence without accepting a CPU fallback.
pub fn validate_worker_mode(row: &Value, logs: &str, gpu: bool) -> Result<TrainingResult> {
    let events: Vec<Value> = logs
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let trained: Vec<&Value> = events
        .iter()
        .filter(|e| e["event"] == "training_complete")
        .collect();
    if trained.len() != 1 || !events.iter().any(|e| e["event"] == "metric_delivered") {
        return Err(invalid(
            "Worker logs do not prove training and acknowledged metric delivery",
        ));
    }
    let result: TrainingResult = serde_json::from_value(trained[0].clone())?;
    let loss = row["val_loss"]
        .as_f64()
        .ok_or_else(|| invalid("Invalid row objective"))?;
    let rate = row["learning_rate"]
        .as_f64()
        .ok_or_else(|| invalid("Invalid row rate"))?;
    if !loss.is_finite()
        || !result.val_loss.is_finite()
        || (result.val_loss - loss).abs() > 1e-12 * loss.abs().max(1.)
        || result.learning_rate != rate
    {
        return Err(invalid(
            "Reported objective/configuration differs from measured model training",
        ));
    }
    let execution_valid = if gpu {
        result.execution == "GPU, Rust CUDA; device training verified"
            && result.gpu.as_ref().is_some_and(|p| {
                !p.device_name.is_empty()
                    && p.compute_capability[0] > 0
                    && p.kernel_launches == 1
                    && p.cuda_elapsed_ms.is_finite()
                    && p.cuda_elapsed_ms > 0.
                    && p.cpu_reference_max_abs_error.is_finite()
                    && (0.0..=1e-10).contains(&p.cpu_reference_max_abs_error)
            })
    } else {
        result.execution == "CPU, Rust; no GPU or simulated score" && result.gpu.is_none()
    };
    if !execution_valid
        || result.epochs != 40
        || result.history.len() != 40
        || !result.weight.is_finite()
        || !result.bias.is_finite()
        || result
            .history
            .iter()
            .enumerate()
            .any(|(i, h)| h.epoch != i + 1 || !h.val_loss.is_finite())
        || result
            .history
            .last()
            .is_none_or(|h| h.val_loss != result.val_loss)
    {
        return Err(invalid("Expected a complete finite Rust training history"));
    }
    Ok(result)
}
/// Submit/watch a typed OptimizationJob, collect logs and independently verify it.
pub fn run(root: &Path, options: &RunOptions) -> Result<Value> {
    if options.timeout.is_zero() {
        return Err(invalid("Timeout must be positive"));
    }
    fs::create_dir_all(&options.output)?;
    let name = if let Some(existing) = &options.existing {
        if !dns_name(existing) {
            return Err(invalid("Invalid existing Kubernetes resource name"));
        }
        existing.clone()
    } else {
        let runtime = kube_json(
            &options.kubeconfig,
            options.gpu,
            &[
                "get",
                "clustertrainingruntime",
                runtime_name(options.gpu),
                "-o",
                "json",
            ],
        )?;
        if runtime.pointer("/spec/template/spec/replicatedJobs/0/template/spec/template/spec/containers/0/image").and_then(Value::as_str) != Some(training_image(options.gpu)) {
            return Err(invalid("Rust training runtime is not installed; run the start command first"));
        }
        let path = root.join("demo/optimizationjob.yaml");
        let path = path
            .to_str()
            .ok_or_else(|| invalid("Non-UTF8 manifest path"))?;
        let mut manifest = kube_json(
            &options.kubeconfig,
            options.gpu,
            &["create", "--dry-run=client", "-f", path, "-o", "json"],
        )?;
        let name = new_job_name();
        manifest["metadata"]["name"] = json!(name);
        manifest["spec"]["trainJobTemplate"]["spec"]["runtimeRef"]["name"] =
            json!(runtime_name(options.gpu));
        kube(
            &options.kubeconfig,
            options.gpu,
            &["create", "-f", "-"],
            Some(&manifest),
        )?;
        name
    };
    println!("Watching real OptimizationJob {name}");
    let start = Instant::now();
    let mut previous = String::new();
    let (job, trials) = loop {
        if start.elapsed() >= options.timeout {
            return Err(invalid("OptimizationJob exceeded completion deadline"));
        }
        let job = kube_json(
            &options.kubeconfig,
            options.gpu,
            &[
                "get",
                "optimizationjob",
                &name,
                "-n",
                NAMESPACE,
                "-o",
                "json",
            ],
        )?;
        let label = format!("trainer.kubeflow.org/optimization-job={name}");
        let trial_list = kube_json(
            &options.kubeconfig,
            options.gpu,
            &[
                "get",
                "trainjobs",
                "-n",
                NAMESPACE,
                "-l",
                &label,
                "-o",
                "json",
            ],
        )?;
        let trials = array(&trial_list["items"])?;
        let progress = format!(
            "{} trials; conditions {}",
            trials.len(),
            job["status"]["conditions"]
        );
        if progress != previous {
            println!("{progress}");
            previous = progress;
        }
        if condition(&job, "Failed") {
            save_json(&options.output, "failed-job.json", &job)?;
            return Err(invalid("OptimizationJob failed; inspect failed-job.json"));
        }
        if condition(&job, "Complete") {
            break (job, trials.clone());
        }
        thread::sleep(Duration::from_secs(2));
    };
    let mut report = summarize(&job, &trials)?;
    let pods = kube_json(
        &options.kubeconfig,
        options.gpu,
        &["get", "pods", "-n", NAMESPACE, "-o", "json"],
    )?;
    let rows = report["trials"]
        .as_array_mut()
        .ok_or_else(|| invalid("Missing report rows"))?;
    for row in rows.iter_mut() {
        let trial_name = string(row, "/name")?.to_owned();
        if !dns_name(&trial_name) {
            return Err(invalid("Invalid trial name"));
        }
        let prefix = format!("{trial_name}-");
        let matched: Vec<&Value> = array(&pods["items"])?
            .iter()
            .filter(|p| {
                p["metadata"]["name"]
                    .as_str()
                    .is_some_and(|n| n.starts_with(&prefix))
            })
            .collect();
        if matched.len() != 1 || matched[0]["status"]["phase"] != "Succeeded" {
            return Err(invalid(
                "Expected exactly one successful real training Pod per trial",
            ));
        }
        let pod = string(matched[0], "/metadata/name")?;
        let logs = kube(
            &options.kubeconfig,
            options.gpu,
            &["logs", pod, "-n", NAMESPACE, "-c", "node"],
            None,
        )?;
        save(&options.output, &format!("{trial_name}.log"), &logs)?;
        if options.gpu
            && matched[0]
                .pointer("/spec/containers/0/resources/limits/nvidia.com~1gpu")
                .and_then(Value::as_str)
                != Some("1")
        {
            return Err(invalid(
                "GPU evidence requires an actual Pod GPU resource allocation",
            ));
        }
        let trained = validate_worker_mode(row, &logs, options.gpu)?;
        if options.gpu {
            row["gpu"] = serde_json::to_value(&trained.gpu)?;
            row["gpu_resource_limit"] = json!(1);
        }
        row["pod"] = json!(pod);
        row["history"] = serde_json::to_value(trained.history)?;
        row["weight"] = json!(trained.weight);
        row["bias"] = json!(trained.bias);
        row["metric_delivered"] = json!(true);
    }
    report["best"] = report["trials"][0].clone();
    report["recorded_at"] = json!(chrono::Utc::now().to_rfc3339());
    report["mode"] = json!(if options.gpu {
        "Real Kubernetes GPU run, Rust CUDA training"
    } else {
        "Real Kubernetes CPU run, Rust workload and tooling"
    });
    report["cluster"] = json!(context(options.gpu));
    report["training"] = json!(
        "64 training + 64 validation samples; 40 gradient descent epochs; y = 2x + 1 + noise"
    );
    report["coverage"] = json!({"real_crd":true,"real_trainjobs":true,"real_optuna":true,"rust_push_metrics":true,"sdk_push_metrics":false,"new_sdk_submission_backend":false,"shared_initializer":false,"gpu":options.gpu});
    let pins: Value = serde_json::from_slice(&fs::read(root.join("upstream.lock.json"))?)?;
    report["source_pins"] = pins["repositories"].clone();
    save_json(&options.output, "optimizationjob.json", &job)?;
    save_json(&options.output, "trainjobs.json", &json!(trials))?;
    save_json(&options.output, "report.json", &report)?;
    crate::plots::write_plots(&report, &options.output)?;
    println!(
        "VERIFIED WINNER: {}\nlearning_rate={}; val_loss={}\nPlots and CSV: {}",
        string(&report, "/best/name")?,
        report["best"]["learning_rate"],
        report["best"]["val_loss"],
        options.output.display()
    );
    Ok(report)
}
fn execute(program: &str, args: &[&str], timeout: u64) -> Result<String> {
    checked(
        Command::new(program).args(args),
        None,
        Duration::from_secs(timeout),
    )
}
fn path_text(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| invalid("Non-UTF8 tool path"))
}
fn pins(root: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(
        root.join("upstream.lock.json"),
    )?)?)
}
fn go_binary() -> String {
    env::var("GO_BINARY").unwrap_or_else(|_| {
        if Path::new("/usr/local/go/bin/go").is_file() {
            "/usr/local/go/bin/go".into()
        } else {
            "go".into()
        }
    })
}

/// Reproduce the existing real cluster using unchanged pinned upstream sources.
pub fn start(root: &Path, options: &RunOptions) -> Result<Value> {
    if options.gpu {
        return crate::gpu_cluster::start(root, options);
    }
    if !cfg!(target_os = "linux") {
        return Err(invalid(
            "Run start under Linux or WSL with Docker, kind, kubectl and Go",
        ));
    }
    let lock = pins(root)?;
    for (name, item) in lock["repositories"]
        .as_object()
        .ok_or_else(|| invalid("Missing source pins"))?
    {
        println!("Verifying pinned {name} checkout");
        verify_checkout(
            &root.join(string(item, "/checkout")?),
            string(item, "/commit")?,
        )?;
    }
    execute("docker", &["info", "--format", "{{.ServerVersion}}"], 30)?;
    let state = root.join(".tools/demo");
    let manager = state.join("build/manager-image");
    let training = state.join("build/rust-training");
    fs::create_dir_all(&manager)?;
    fs::create_dir_all(&training)?;
    let clusters = execute("kind", &["get", "clusters"], 30)?;
    if !clusters.lines().any(|c| c == "optimizationjob-demo") {
        println!("Creating dedicated kind cluster optimizationjob-demo");
        execute(
            "kind",
            &[
                "create",
                "cluster",
                "--name",
                "optimizationjob-demo",
                "--config",
                path_text(&root.join("demo/kind.yaml"))?,
                "--kubeconfig",
                path_text(&options.kubeconfig)?,
                "--wait",
                "120s",
            ],
            240,
        )?;
    }
    let running = execute(
        "docker",
        &[
            "inspect",
            "--format",
            "{{.State.Running}}",
            "optimizationjob-demo-control-plane",
        ],
        30,
    )?;
    if running.trim() != "true" {
        let owner = execute(
            "docker",
            &[
                "inspect",
                "--format",
                "{{index .Config.Labels \"io.x-k8s.kind.cluster\"}}",
                "optimizationjob-demo-control-plane",
            ],
            30,
        )?;
        if owner.trim() != "optimizationjob-demo" {
            return Err(invalid(
                "Existing container does not belong to our dedicated kind cluster",
            ));
        }
        println!("Restarting the stopped dedicated kind node");
        execute(
            "docker",
            &["start", "optimizationjob-demo-control-plane"],
            60,
        )?;
    }
    wait_for(
        Duration::from_secs(120),
        Duration::from_secs(1),
        |remaining| {
            checked(
                Command::new("kubectl")
                    .arg("--kubeconfig")
                    .arg(&options.kubeconfig)
                    .args(["--context", CONTEXT, "get", "nodes", "-o", "name"]),
                None,
                remaining.min(Duration::from_secs(5)),
            )
            .map(|_| ())
        },
    )?;
    checked(
        Command::new("kubectl")
            .arg("--kubeconfig")
            .arg(&options.kubeconfig)
            .args([
                "--context",
                CONTEXT,
                "wait",
                "--for=condition=Ready",
                "nodes",
                "--all",
                "--timeout=120s",
            ]),
        None,
        Duration::from_secs(140),
    )?;
    println!("Building pinned upstream Trainer controller");
    let trainer = root.join(string(&lock["repositories"]["trainer"], "/checkout")?);
    checked(
        Command::new(go_binary())
            .current_dir(trainer)
            .env("CGO_ENABLED", "0")
            .arg("build")
            .arg("-o")
            .arg(manager.join("manager"))
            .arg("./cmd/trainer-controller-manager"),
        None,
        Duration::from_secs(600),
    )?;
    fs::copy(
        root.join("demo/Dockerfile.manager"),
        manager.join("Dockerfile"),
    )?;
    execute(
        "docker",
        &[
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "-t",
            "optimizationjob-demo-manager:b12f968",
            path_text(&manager)?,
        ],
        600,
    )?;
    println!("Packaging this Rust executable as the training image");
    fs::copy(env::current_exe()?, training.join("optimizationjob"))?;
    fs::copy(
        root.join("demo/Dockerfile.training"),
        training.join("Dockerfile"),
    )?;
    execute(
        "docker",
        &[
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "-t",
            TRAINING_IMAGE,
            path_text(&training)?,
        ],
        600,
    )?;
    execute("docker", &["pull", PROVIDER], 180)?;
    execute("docker", &["tag", PROVIDER, PROVIDER_TAG], 30)?;
    let archive = state.join("rust-images-amd64.tar");
    println!("Loading controller, Rust worker and Optuna into kind");
    execute(
        "docker",
        &[
            "image",
            "save",
            "--platform",
            "linux/amd64",
            "-o",
            path_text(&archive)?,
            "optimizationjob-demo-manager:b12f968",
            TRAINING_IMAGE,
            PROVIDER_TAG,
        ],
        180,
    )?;
    execute(
        "kind",
        &[
            "load",
            "image-archive",
            "--name",
            "optimizationjob-demo",
            path_text(&archive)?,
        ],
        240,
    )?;
    let overlay = root.join("demo/deploy");
    kube(
        &options.kubeconfig,
        options.gpu,
        &["apply", "--server-side", "-k", path_text(&overlay)?],
        None,
    )?;
    for deployment in [
        "jobset-controller-manager",
        "kubeflow-trainer-controller-manager",
    ] {
        println!("Waiting for {deployment}");
        checked(
            Command::new("kubectl")
                .arg("--kubeconfig")
                .arg(&options.kubeconfig)
                .args([
                    "--context",
                    CONTEXT,
                    "rollout",
                    "status",
                    &format!("deployment/{deployment}"),
                    "-n",
                    "kubeflow-system",
                    "--timeout=300s",
                ]),
            None,
            Duration::from_secs(320),
        )?;
    }
    kube(
        &options.kubeconfig,
        options.gpu,
        &[
            "apply",
            "--server-side",
            "-f",
            path_text(&root.join("demo/runtime.yaml"))?,
        ],
        None,
    )?;
    run(root, options)
}
/// Execute upstream Go baseline tests after verifying revision and cleanliness.
pub fn smoke(root: &Path, trainer: Option<&Path>, output: &Path) -> Result<()> {
    let lock = pins(root)?;
    let default_trainer = root.join(string(&lock["repositories"]["trainer"], "/checkout")?);
    let trainer = trainer.unwrap_or(&default_trainer);
    verify_checkout(
        trainer,
        string(&lock["repositories"]["trainer"], "/commit")?,
    )?;
    fs::create_dir_all(output)?;
    println!("{}", execute(&go_binary(), &["version"], 30)?.trim());
    for (package, path) in [
        ("optimizationjob", "./pkg/util/optimizationjob"),
        ("controller", "./pkg/controller"),
    ] {
        println!("Testing pinned upstream {path}");
        let result = capture(
            Command::new(go_binary())
                .current_dir(trainer)
                .env("CGO_ENABLED", "0")
                .args(["test", "-json", path, "-count=1"]),
            None,
            Duration::from_secs(600),
        )?;
        save(output, &format!("smoke-{package}.jsonl"), &result.stdout)?;
        save(output, &format!("smoke-{package}.stderr"), &result.stderr)?;
        let events: Vec<Value> = result
            .stdout
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        let passes = events
            .iter()
            .filter(|e| e["Action"] == "pass" && e["Test"].is_string())
            .count();
        let failed = events.iter().any(|e| e["Action"] == "fail");
        if !result.status.success() || passes == 0 || failed {
            return Err(invalid(format!(
                "Upstream {package} tests failed; see saved evidence"
            )));
        }
        println!("{passes} passing test/subtest events; no failures");
    }
    Ok(())
}

/// Keep enough name space for upstream Trial, JobSet and Pod suffixes.
pub fn new_job_name() -> String {
    format!("rust-hpo-{}", chrono::Utc::now().timestamp_millis())
}

/// Select only one of the two dedicated demo contexts.
pub fn context(gpu: bool) -> &'static str {
    if gpu {
        "kind-rubixkube-gpu"
    } else {
        CONTEXT
    }
}
/// Distinct CPU and GPU runtimes preserve the original working CPU setup.
pub fn runtime_name(gpu: bool) -> &'static str {
    if gpu {
        "rubixkube-demo-gpu"
    } else {
        "optimizationjob-demo-cpu"
    }
}
/// A CUDA image can never be mistaken for the CPU image.
pub fn training_image(gpu: bool) -> &'static str {
    if gpu {
        "rubixkube-training:cuda"
    } else {
        TRAINING_IMAGE
    }
}
