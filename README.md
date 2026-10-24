# OptimizationJob — Rust integration demo

This repository contains Rust training, authenticated metric delivery, Kubernetes demo orchestration, baseline tooling, report validation and tests. Upstream Kubeflow Trainer, SDK and Katib remain unchanged sibling dependencies pinned in `upstream.lock.json`.

The local demo submits a real `trainer.kubeflow.org/v1alpha1` OptimizationJob. Four actual CPU training Pods fit a linear model and report held-out loss. The upstream controller and Optuna choose the trials and winner. See [demo/README.md](demo/README.md) for the full walkthrough.

## Source map

| File | Responsibility |
|---|---|
| `src/main.rs` | CLI: `train`, `run`, `start`, `smoke`, `serve` |
| `src/training.rs` | Deterministic model fitting and validation history |
| `src/metrics.rs` | HTTPS metric push with projected CA and bearer token |
| `src/process.rs` | Bounded external commands, source cleanliness/revision checks |
| `src/report.rs` | Independent objective ranking and winner validation |
| `src/orchestrate.rs` | Cluster installation, submission, evidence collection, baseline |
| `src/serve.rs` | Loopback report server with path containment |
| `tests/*.rs` | Training parity, reports, TLS, Git guards, process deadlines, CLI/server |
| `demo/fixtures/` | Original dataset and pre-migration Python numerical references |

YAML, JSON, Dockerfiles and the interactive HTML/CSS/JavaScript report remain in their native formats. There is no owned Python or Bash implementation. Python/Go inside unchanged upstream dependencies were deliberately retained.

## Build and check

With Rust 1.90.0 and Git installed:

```text
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo build --locked --release
cargo run --locked -- train --local --learning-rate 0.1
```

The full demo requires Linux/WSL, Docker, kind, kubectl, Go compatible with the pinned Trainer revision, and the three sibling source checkouts. It does not need a local Python installation. `GO_BINARY` can select the Go executable. In the prepared WSL environment it defaults to `/usr/local/go/bin/go`.

For this Windows workspace, a compiled Linux executable is staged at `.tools/rust/optimizationjob`:

```powershell
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/rust/optimizationjob start
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/rust/optimizationjob run
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/rust/optimizationjob smoke
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/rust/optimizationjob serve
```

The executable discovers the repository from its own path. Supply `--root` when running a binary installed elsewhere. `serve` defaults to [http://127.0.0.1:8765/](http://127.0.0.1:8765/).

## Isolated Docker build without installing Rust

Run these commands inside WSL from `/mnt/h/GSocProjects/optimizationjob`. The official Rust build image is pinned by digest. Named volumes hold Cargo downloads and compilation output; the source remains on H:.

```bash
docker run --rm -v "$PWD:/work" -v optimizationjob-rust-target:/target -v optimizationjob-rust-cargo:/usr/local/cargo/registry -e CARGO_TARGET_DIR=/target -w /work rust@sha256:cfb3f582db21e4b4168bffa96397db118d288f1c55026cf016911e147476184e cargo test --locked
docker run --rm -v "$PWD:/work" -v optimizationjob-rust-target:/target -v optimizationjob-rust-cargo:/usr/local/cargo/registry -e CARGO_TARGET_DIR=/target -w /work rust@sha256:cfb3f582db21e4b4168bffa96397db118d288f1c55026cf016911e147476184e cargo build --locked --release
mkdir -p .tools/rust
docker run --rm -v "$PWD:/work" -v optimizationjob-rust-target:/target rust@sha256:cfb3f582db21e4b4168bffa96397db118d288f1c55026cf016911e147476184e cp /target/release/optimizationjob /work/.tools/rust/optimizationjob
.tools/rust/optimizationjob start
```

The `start` command packages the executable you invoke as the non-root training image. Use a release Linux x86_64 build for the pinned amd64 cluster. Kubernetes credentials, local executables, environments and raw evidence stay ignored. Plans live in `../optimizationjob-plan`; current Rust results live in `../optimizationjob-evidence/runs/rust/`. Previous Python demo evidence is preserved separately in `runs/demo/`.

## Scope

This is a real integration demonstration, not a new Rust implementation of the upstream OptimizationJob controller or SDK. GPU execution, shared initialization, a new OptimizerClient backend and production crash-recovery guarantees remain outside this migration. The metrics client implements Trainer's existing protocol directly; the Python SDK is not executed by the Rust worker.
