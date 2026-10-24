# RubixKube: real CUDA training on local WSL Kubernetes

This optional backend executes training on the NVIDIA GPU. It does not fall back to CPU if CUDA is unavailable. The CPU backend and original demo remain available unchanged.

## Build and run

With Rust 1.90.0 on Linux/WSL:

```text
cargo test --locked --features cuda
cargo build --locked --release --features cuda
cargo run --locked --release --features cuda -- start --gpu
cargo run --locked --release --features cuda -- run --gpu
cargo run --locked --release --features cuda -- run --gpu --existing GPU_JOB_NAME
```

`start --gpu` creates/uses only `rubixkube-gpu`, with context `kind-rubixkube-gpu` and private kubeconfig `.tools/gpu/kubeconfig`. It verifies pinned upstream source, builds the controller and CUDA worker, loads images, installs the official NVIDIA device plugin and unchanged Trainer/JobSet, waits for one allocatable GPU, then runs four sequential trials. The demo requires `/dev/dxg` and `/usr/lib/wsl`, Docker, kind, kubectl and compatible Go; it targets this Windows/WSL setup. It does not install Linux display drivers or modify the host Docker daemon.

A compiled CUDA-enabled executable is staged at `.tools/gpu/rubixkube` in the prepared workspace:

```powershell
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/gpu/rubixkube start --gpu
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/gpu/rubixkube run --gpu
wsl -d Ubuntu-22.04 -- /mnt/h/GSocProjects/optimizationjob/.tools/gpu/rubixkube plot --report /mnt/h/GSocProjects/optimizationjob-evidence/runs/gpu/reports/report.json
```

To use the existing isolated Docker Rust toolchain, run the root README Docker commands with `--features cuda` added to `cargo build` and `cargo test`, and export the resulting release executable to `.tools/gpu/rubixkube`. Stop an executable before overwriting it or stage a new path; in-place overwrite of a running Linux executable fails.

## What actually runs on the GPU

`src/gpu.rs` uses cudarc's CUDA Driver and NVRTC interfaces. Rust transfers the fixed 128 examples to device memory and compiles a CUDA kernel at runtime. One 64-thread block performs all 40 epochs:

1. Each thread computes one training example's error and gradient contribution.
2. Device-side reduction updates weight and bias.
3. Each thread computes one held-out example's squared error.
4. Device-side reduction writes the epoch's validation MSE.

CUDA synchronization completes the kernel before any result is reported. Rust copies the coefficients and full loss history back to host memory. It independently runs the CPU reference and rejects differences above `1e-10`, nonfinite results or missing positive CUDA-event timing. CUDA errors cause failure, without replacing the results with CPU values.

The embedded kernel uses CUDA C syntax; application, orchestration and verification code remain Rust. The CPU reference is a verification calculation performed after the independent GPU computation, not a fallback training path.

## Evidence

Each successful worker log contains `training_complete` with:

- `execution: "GPU, Rust CUDA; device training verified"`
- `gpu.device_name` and `compute_capability`
- `gpu.kernel_launches: 1` (one kernel performs every epoch)
- Positive `gpu.cuda_elapsed_ms` between CUDA events
- `gpu.cpu_reference_max_abs_error`
- Actual GPU-derived coefficients and the complete validation history

A separate `metric_delivered` event proves Trainer acknowledged final metrics. The runner also checks the actual successful Pod requested `nvidia.com/gpu: 1`; a CPU execution string or missing GPU proof is rejected in GPU mode. Saved report JSON includes the GPU evidence per trial and `coverage.gpu: true` only after those checks.

`../optimizationjob-evidence/runs/gpu/` holds external test outputs, image identities and real cluster evidence. `reports/` holds the checked result, resources, worker logs, plain PNG/SVG graphs, numeric CSV and a text summary. `plot --report PATH` regenerates graphs from the saved data without accessing the cluster. No web app or HTTP server is included. Kubeconfig and host driver files stay outside Git. Example plots and numeric results are committed in `demo/results/gpu/`.

## Hardware tests

The default suite skips the hardware-dependent test unless explicitly requested. In a CUDA-enabled environment with actual device access and CUDA 12.8 NVRTC libraries, execute:

```text
cargo test --locked --features cuda --test gpu -- --ignored --nocapture
```

The test independently executes every candidate learning rate on CUDA and compares with CPU. This test was also run in the official CUDA container with `/dev/dxg` supplied and `/usr/lib/wsl` mounted read-only. A read-only-root, non-root container with dropped capabilities was tested separately.

## Scope and security

This is a single-local-GPU WSL development demo. The official device plugin needs privileged access to discover/pass the WSL GPU; it runs only in the dedicated GPU cluster. Training Pods remain non-root, read-only, with no privilege escalation and dropped capabilities. WSL exposes GPUs through `/dev/dxg`; this configuration does not claim production multi-GPU or tenant isolation. Driver directories are mounted read-only and not copied into the repository.

The GPU runtime image and official plugin are pinned by digest. CUDA toolkit libraries are in the container; the Windows/WSL driver remains host-provided. The same existing cgroup-v1 compatibility setting is retained for kind. GPU kernels and CUDA events prove execution; detecting a GPU or observing `nvidia-smi` alone is not treated as sufficient proof. This tiny workload is a correctness demonstration, not a GPU speed benchmark.

To remove only this disposable GPU cluster after inspection:

```powershell
wsl -d Ubuntu-22.04 -- kind delete cluster --name rubixkube-gpu
```

The separate CPU cluster is unaffected.
