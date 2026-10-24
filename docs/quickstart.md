# Local shared-data GPU quickstart

Use the prepared WSL workspace (Rust 1.90, Go, Docker, kind, kubectl, upstream clones and the existing NVIDIA Windows driver). Run from the repository root:

```text
cargo install --path . --locked --features cuda --bin rubixkube
rubixkube start --gpu
```

The helper installs pinned upstream components in the dedicated `rubixkube-gpu` cluster; it leaves the separate CPU cluster alone. Build/load the packaged worker used by the configurable examples:

```text
docker build --target cuda --platform linux/amd64 -t rubixkube-worker:0.2 .
docker image save --platform linux/amd64 -o /tmp/rubixkube-worker.tar rubixkube-worker:0.2
kind load image-archive --name rubixkube-gpu /tmp/rubixkube-worker.tar
rubixkube validate --config examples/shared-gpu.json
rubixkube doctor --config examples/shared-gpu.json --kubeconfig .tools/gpu/kubeconfig --context kind-rubixkube-gpu
rubixkube submit --config examples/shared-gpu.json --name my-gpu-fit --wait --kubeconfig .tools/gpu/kubeconfig --context kind-rubixkube-gpu --output runs/my-gpu-fit
```

`examples/shared-gpu.json` prepares a 256-row CSV for `y = 3x + 0.5` once, then optimizes learning rate and epoch count. Four trials share one read-only PVC and report a custom `validation_error` metric. Parallelism two with one physical GPU naturally queues workers. The example uses a pinned BusyBox initializer; your own initializer can download real datasets/model files instead.

Inspect `runs/my-gpu-fit/summary.txt`, `results.csv`, `objective.png` and the raw job/Pod/log evidence. `watch` and `results` resume the same job without creating more trials.

For your own model, replace `workload.image`, command/args and parameter/metric definitions. The image must report the final metric before successful exit. For an existing remote cluster you do not need the local setup helper or Go/Python clones; use its installed runtime and accessible registry image with explicit connection arguments.

Shared data remains until you explicitly purge it:

```text
rubixkube cancel my-gpu-fit --namespace optimizationjob-demo --kubeconfig .tools/gpu/kubeconfig --context kind-rubixkube-gpu
rubixkube cancel my-gpu-fit --purge-shared-data --namespace optimizationjob-demo --kubeconfig .tools/gpu/kubeconfig --context kind-rubixkube-gpu
```
