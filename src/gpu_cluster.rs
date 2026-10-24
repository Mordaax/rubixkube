//! Reproducible single-GPU WSL kind setup, isolated from the CPU demo cluster.
use crate::{
    invalid,
    orchestrate::{kube, run, RunOptions},
    process::{checked, verify_checkout, wait_for},
    report::string,
    Result,
};
use serde_json::Value;
use std::{env, fs, path::Path, process::Command, time::Duration};
const CLUSTER: &str = "rubixkube-gpu";
const NODE: &str = "rubixkube-gpu-control-plane";
const PLUGIN: &str = "nvcr.io/nvidia/k8s-device-plugin:v0.20.1";
const PROVIDER: &str = "ghcr.io/kubeflow/katib/suggestion-optuna:v0.19.0";
const MANAGER: &str = "optimizationjob-demo-manager:b12f968";
fn text(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| invalid("Non-UTF8 path"))
}
fn exec(program: &str, args: &[&str], timeout: u64) -> Result<String> {
    checked(
        Command::new(program).args(args),
        None,
        Duration::from_secs(timeout),
    )
}
fn kubectl(options: &RunOptions, args: &[&str], timeout: u64) -> Result<String> {
    checked(
        Command::new("kubectl")
            .arg("--kubeconfig")
            .arg(&options.kubeconfig)
            .args(["--context", "kind-rubixkube-gpu"])
            .args(args),
        None,
        Duration::from_secs(timeout),
    )
}
/// Build and run CUDA trials in the dedicated WSL GPU cluster.
pub fn start(root: &Path, options: &RunOptions) -> Result<Value> {
    if !cfg!(feature = "cuda") {
        return Err(invalid("GPU setup requires a build with --features cuda"));
    }
    if !Path::new("/dev/dxg").exists() || !Path::new("/usr/lib/wsl/lib/libcuda.so.1").exists() {
        return Err(invalid("This cluster setup targets WSL GPU passthrough; /dev/dxg and WSL CUDA driver libraries are required"));
    }
    let lock: Value = serde_json::from_slice(&fs::read(root.join("upstream.lock.json"))?)?;
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
    let state = root.join(".tools/gpu");
    let build = state.join("build");
    let manager = state.join("manager");
    fs::create_dir_all(&build)?;
    fs::create_dir_all(&manager)?;
    let clusters = exec("kind", &["get", "clusters"], 30)?;
    if !clusters.lines().any(|name| name == CLUSTER) {
        println!("Creating separate WSL GPU cluster {CLUSTER}");
        exec(
            "kind",
            &[
                "create",
                "cluster",
                "--name",
                CLUSTER,
                "--config",
                text(&root.join("demo/gpu/kind.yaml"))?,
                "--kubeconfig",
                text(&options.kubeconfig)?,
                "--wait",
                "120s",
            ],
            240,
        )?;
    }
    let owner = exec(
        "docker",
        &[
            "inspect",
            "--format",
            "{{index .Config.Labels \"io.x-k8s.kind.cluster\"}}",
            NODE,
        ],
        30,
    )?;
    if owner.trim() != CLUSTER {
        return Err(invalid(
            "Existing GPU node does not belong to our dedicated cluster",
        ));
    }
    if exec(
        "docker",
        &["inspect", "--format", "{{.State.Running}}", NODE],
        30,
    )?
    .trim()
        != "true"
    {
        exec("docker", &["start", NODE], 60)?;
    }
    wait_for(
        Duration::from_secs(120),
        Duration::from_secs(1),
        |remaining| {
            checked(
                Command::new("kubectl")
                    .arg("--kubeconfig")
                    .arg(&options.kubeconfig)
                    .args([
                        "--context",
                        "kind-rubixkube-gpu",
                        "get",
                        "nodes",
                        "-o",
                        "name",
                    ]),
                None,
                remaining.min(Duration::from_secs(5)),
            )
            .map(|_| ())
        },
    )?;
    kubectl(
        options,
        &[
            "wait",
            "--for=condition=Ready",
            "nodes",
            "--all",
            "--timeout=120s",
        ],
        140,
    )?;
    println!("Building pinned upstream controller and CUDA Rust worker image");
    let go = env::var("GO_BINARY").unwrap_or_else(|_| "/usr/local/go/bin/go".into());
    checked(
        Command::new(go)
            .current_dir(root.join(string(&lock["repositories"]["trainer"], "/checkout")?))
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
    exec(
        "docker",
        &[
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "-t",
            MANAGER,
            text(&manager)?,
        ],
        300,
    )?;
    fs::copy(env::current_exe()?, build.join("optimizationjob"))?;
    fs::copy(
        root.join("demo/gpu/Dockerfile.training"),
        build.join("Dockerfile"),
    )?;
    exec(
        "docker",
        &[
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "-t",
            "rubixkube-training:cuda",
            text(&build)?,
        ],
        600,
    )?;
    // Pull immutable provider/plugin versions before exporting the amd64 images.
    exec("docker",&["pull","ghcr.io/kubeflow/katib/suggestion-optuna@sha256:04a71f288d04251835846b87d18a9fcf816eb59003d282555dc3d94412c879c7"],180)?;
    exec("docker",&["tag","ghcr.io/kubeflow/katib/suggestion-optuna@sha256:04a71f288d04251835846b87d18a9fcf816eb59003d282555dc3d94412c879c7",PROVIDER],30)?;
    exec("docker",&["pull","nvcr.io/nvidia/k8s-device-plugin@sha256:27c1b2553a690ca3d29889dcd61df97525d46f58272c0be8dc63c2b18f120075"],180)?;
    exec("docker",&["tag","nvcr.io/nvidia/k8s-device-plugin@sha256:27c1b2553a690ca3d29889dcd61df97525d46f58272c0be8dc63c2b18f120075",PLUGIN],30)?;
    println!("Loading CUDA/controller/provider/device-plugin images into {CLUSTER}");
    let archive = state.join("images-amd64.tar");
    exec(
        "docker",
        &[
            "image",
            "save",
            "--platform",
            "linux/amd64",
            "-o",
            text(&archive)?,
            MANAGER,
            "rubixkube-training:cuda",
            PROVIDER,
            PLUGIN,
        ],
        300,
    )?;
    exec(
        "kind",
        &["load", "image-archive", "--name", CLUSTER, text(&archive)?],
        360,
    )?;
    kube(
        &options.kubeconfig,
        true,
        &[
            "apply",
            "-f",
            text(&root.join("demo/gpu/device-plugin.yaml"))?,
        ],
        None,
    )?;
    kubectl(
        options,
        &[
            "-n",
            "kube-system",
            "rollout",
            "status",
            "daemonset/rubixkube-nvidia-device-plugin",
            "--timeout=120s",
        ],
        140,
    )?;
    wait_for(
        Duration::from_secs(120),
        Duration::from_secs(1),
        |remaining| {
            let nodes = checked(
                Command::new("kubectl")
                    .arg("--kubeconfig")
                    .arg(&options.kubeconfig)
                    .args([
                        "--context",
                        "kind-rubixkube-gpu",
                        "get",
                        "nodes",
                        "-o",
                        "json",
                    ]),
                None,
                remaining.min(Duration::from_secs(5)),
            )?;
            let nodes: Value = serde_json::from_str(&nodes)?;
            if nodes["items"].as_array().is_some_and(|items| {
                items.iter().any(|n| {
                    n.pointer("/status/allocatable/nvidia.com~1gpu")
                        .and_then(Value::as_str)
                        == Some("1")
                })
            }) {
                Ok(())
            } else {
                Err(invalid("Waiting for one advertised nvidia.com/gpu"))
            }
        },
    )?;
    println!("Kubernetes advertises one allocatable GPU; installing unchanged Trainer/JobSet");
    kube(
        &options.kubeconfig,
        true,
        &[
            "apply",
            "--server-side",
            "-k",
            text(&root.join("demo/deploy"))?,
        ],
        None,
    )?;
    for deployment in [
        "jobset-controller-manager",
        "kubeflow-trainer-controller-manager",
    ] {
        kubectl(
            options,
            &[
                "-n",
                "kubeflow-system",
                "rollout",
                "status",
                &format!("deployment/{deployment}"),
                "--timeout=300s",
            ],
            320,
        )?;
    }
    kube(
        &options.kubeconfig,
        true,
        &[
            "apply",
            "--server-side",
            "-f",
            text(&root.join("demo/gpu/runtime.yaml"))?,
        ],
        None,
    )?;
    run(root, options)
}
