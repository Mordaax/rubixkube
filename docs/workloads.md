# Training-container contract

RubixKube can optimize a user-supplied container rather than only its built-in model. Build your image with the model, framework and data loader it needs. Set `workload.image`, and optionally its command/args, in the JSON configuration. Provide an existing compatible Kubeflow Trainer runtime. v0.2 collection expects one successful trainer Pod/container named `node` per trial.

1. Read sampled values from `KUBEFLOW_TRAINER_OPT_<parameter>`.
2. Train your model and measure the configured objective.
3. Push the final finite metric through Trainer's status API and wait for acknowledgement.
4. Exit successfully only after reporting succeeds; return nonzero on training/report failure.

The `TrainJobStatus` feature gate must be enabled in the controller. It projects `KUBEFLOW_TRAINER_SERVER_URL`, `KUBEFLOW_TRAINER_SERVER_CA_CERT` and `KUBEFLOW_TRAINER_SERVER_TOKEN`. Do not override these or `KUBEFLOW_` environment names in configuration. Never copy projected token values into logs, config or images.

Three reporting options:

- Include the RubixKube binary and invoke `rubixkube report --metric validation_error --value 0.12`.
- From Rust, call `rubixkube::metrics::deliver_metric(&MetricsConfig::from_env()?, "validation_error", value)`.
- Use the upstream Kubeflow SDK's `update_trainjob_status` and check its success result.

A GPU resource request only proves allocation. To use `require_cuda_proof`, the worker must emit the built-in `training_complete` evidence contract: matching `objective_metric`, finite `val_loss`, full epoch history, coefficients, GPU name/compute capability, completed kernel count, positive CUDA event timing and CPU-reference parity error at most `1e-10`. This is a worker instrumentation contract, not a hardware attestation scheme for untrusted images. Other GPU workloads may report ordinary metrics with that option disabled; results will not claim CUDA training was verified.

## Built-in CSV workload

```text
rubixkube train --local --dataset data.csv --learning-rate 0.1 --epochs 100
rubixkube train --local --dataset data.csv --learning-rate 0.1 --epochs 100 --device cuda
```

CSV is two numeric columns `x,y` (optional header), 4..1,000,000 finite rows and at most 32 MiB. The first half trains and the second half validates. Inputs should be normalized; the simple linear model accepts learning rates in `(0, 0.2]` and 1..10000 epochs, and fails on divergence. This is not a general neural-network training library. Use your own model image for other algorithms.

Inside a TrainJob, omit `--local` to require acknowledged metric delivery. `learning_rate` and `epochs` can be searched through the corresponding injected environment values. Use `--metric NAME` to report the measured MSE under another configured name. Shared-data examples pass `--dataset /dataset/training.csv`; their initializer runs before the search and the file is mounted read-only.

## Images

```text
docker build --target cpu -t rubixkube-worker:0.2-cpu .
docker build --target cuda -t rubixkube-worker:0.2-cuda .
```

These images enter the `rubixkube` CLI. For built-in training, use a workload command such as `["/app/rubixkube", "train"]`, adding `"--device", "cuda"` for CUDA, and dataset/metric arguments as needed. Supply a compatible NVIDIA runtime/device on native Linux, or use the documented WSL setup. Push your image to a registry your cluster can pull, or load it into kind for local use. Kubernetes image pull policy comes from the chosen runtime.

The client binary requires `kubectl` for cluster operations; the training/metric-reporting image does not bundle it. General lifecycle commands do not require the demo's sibling Go/Python source checkouts. Namespace, runtime, service account, RBAC and storage configuration are cluster prerequisites.
