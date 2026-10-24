# Real Kubernetes demonstration with Rust

## What runs

1. Our Rust launcher verifies the pinned, clean upstream checkouts and prepares the dedicated `optimizationjob-demo` kind cluster.
2. The unchanged Go Trainer controller and JobSet run in Kubernetes. The unchanged Katib Optuna provider supplies Random search suggestions.
3. Our Rust runner submits a typed OptimizationJob with four trials, parallelism one, seed 42, and candidate learning rates `0.005`, `0.02`, `0.05`, `0.1`.
4. Each real training Pod executes our Rust binary. It fits `prediction = weight*x + bias` for 40 full-batch gradient-descent epochs.
5. The worker computes validation MSE on 64 held-out examples, then uses our Rust HTTPS client to push the final `val_loss` to Trainer. The projected CA, hostname and bearer token are verified; HTTP 200 is required before the worker reports success.
6. The controller chooses the lowest measured loss. Our Rust runner independently checks the trial budget, completion conditions, finite metrics, winner ordering, successful Pods, complete Rust training history and acknowledgement event before saving the report.

Random search can repeat a candidate. Four trials do not guarantee that every candidate is sampled. The deterministic original dataset is stored in `fixtures/dataset.json`; `fixtures/parity.json` records the pre-migration Python results. Rust tests compare every epoch, coefficient and final loss to within an absolute `1e-12` tolerance. The rate `0.1` yields validation MSE approximately `0.0073170055` on that dataset.

The dataset is embedded in the Rust executable. It is shared as source data, not through a Kubernetes shared initializer or shared artifact volume.

## Run it

Build with Rust 1.90.0 using the root README instructions. Inside the repository on Linux/WSL:

```text
cargo run --locked --release -- start
cargo run --locked --release -- run
cargo run --locked --release -- run --existing RUST_JOB_NAME
cargo run --locked --release -- serve
```

`start` builds/reloads images and installs the runtime before creating a new job. `run` only submits or inspects a job in the existing installation, and refuses to submit if the Rust worker runtime is absent. Both accept `--kubeconfig`, `--output`, `--timeout` and `--existing`. For `start --existing`, the installation is refreshed before that existing job is checked. Old Python jobs are preserved but cannot be represented as verified Rust execution.

`train --local --learning-rate 0.1` fits a model without Kubernetes and prints its actual loss history. It does not claim metric delivery. Inside a TrainJob, plain `train` reads the injected learning-rate environment variable and requires acknowledged delivery. The Kubernetes demonstration uses 40 epochs; custom epoch counts are available with `--local`.

## Credentials and protocol

The worker consumes the existing Trainer variables:

- `KUBEFLOW_TRAINER_OPT_learning_rate`
- `KUBEFLOW_TRAINER_SERVER_URL`
- `KUBEFLOW_TRAINER_SERVER_CA_CERT`
- `KUBEFLOW_TRAINER_SERVER_TOKEN`

The HTTPS body has `trainerStatus.lastUpdatedTime`, `progressPercentage: 100`, and `metrics: [{name: "val_loss", value: "..."}]`. Tokens are reread for each attempt. Redirects and proxies are disabled. The request timeout is five seconds, with up to eight attempts and 5.5 seconds between failures. Only a final objective is sent, so repeated per-epoch updates and SDK throttling are unnecessary. Token values are never written into logs or reports.

This is a protocol-compatible Rust client for this workload, not a port of every Python SDK API. The exact original contract comes from the pinned SDK `kubeflow/trainer/backends/kubernetes/utils.py` and Trainer status-server sources.

## Evidence and browser walkthrough

By default Rust writes `../optimizationjob-evidence/runs/rust/reports/`:

- `report.json`: independently checked winner, trial metrics, coefficients, histories and coverage flags.
- `optimizationjob.json` and `trainjobs.json`: actual observed resources.
- One worker `.log` per trial, including `training_complete` and `metric_delivered` events.
- `index.html`: interactive saved results; selecting a trial displays its measured learning curve.

The saved page is not a live dashboard. `serve` binds only `127.0.0.1`, defaults to port 8765, and serves this report directory. It rejects traversal and symlink escapes. Supply `--port 8766` if an older report server is still using 8765. Kubeconfig is stored separately in `.tools/demo/kubeconfig` and is never served by default. Preserve a report run by choosing another `--output` directory before running again.

## Compatibility and reproducibility

The dedicated cluster/context and kubeconfig isolate this demo from other clusters. The launcher can restart its stopped single kind node after checking its kind cluster label. It does not start unrelated containers. If WSL shuts down when no foreground task is active, keep a WSL terminal open while inspecting the cluster.

The pinned kind configuration keeps the existing Kubernetes 1.37/cgroup-v1 compatibility setting (`failCgroupV1: false`). Controller startup retains its certificate startup probe. Images are exported/imported only for linux/amd64 to avoid partial multi-platform OCI indexes.

Rust toolchain, Cargo.lock, upstream revisions, node image, Optuna image and Rust worker runtime base are pinned. `Dockerfile.training` copies the invoked Linux executable into the pinned distroless C/C++ runtime; the worker is Rust and requires no Python runtime. The manager Dockerfile retains its original distroless base tag, so this demo does not claim a fully hermetic rebuild of all dependencies.

Worker Pods run as UID 65532 with no privilege escalation, a read-only root filesystem, dropped capabilities and bounded CPU/memory. The existing NetworkPolicy is retained, but kind's default CNI does not enforce it. The upstream provider's internal gRPC remains unchanged. Use the cluster as a trusted local development demo.

## Checks and cleanup

```text
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo run --locked --release -- smoke
```

The Rust suite covers numerical parity, objective selection/refusal paths, TLS authentication and certificate rejection, finite retries, corrupt/dirty Git source checks, process deadlines/large output, CLI and contained serving. `smoke` executes the actual pinned upstream Go packages and saves structured logs outside Git. Real Kubernetes acceptance is a separate executed check.

This page documents the CPU path. An optional CUDA backend and separate GPU Kubernetes demo are documented in [gpu/README.md](gpu/README.md). Shared initialization, Grid search, controller/provider crash recovery and a new SDK OptimizationJob backend remain unfinished. It does not convert the upstream Go/Python projects or claim production readiness.

To remove only this explicitly named disposable cluster after inspection:

```powershell
wsl -d Ubuntu-22.04 -- kind delete cluster --name optimizationjob-demo
```

Stop the Rust `serve` process separately when finished. No cleanup is automatic.
