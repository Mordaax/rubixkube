# Configuration reference

The file is JSON with `version: 1`; unknown fields are errors. `rubixkube init --gpu` writes an editable example. `validate --config FILE` is offline; add `--server --kubeconfig FILE --context NAME` for admission validation. `submit --dry-run` prints the rendered OptimizationJob and does not initialize data or create resources.

| Field | Meaning |
|---|---|
| `namespace` | Existing Kubernetes namespace |
| `runtime.name`, `runtime.kind` | Existing ClusterTrainingRuntime (default kind) or namespaced TrainingRuntime |
| `workload.image` | Your worker container image |
| `workload.command`, `args` | Optional arrays overriding image/runtime execution |
| `workload.env` | Literal non-secret environment values |
| `workload.secret_env` | Map from env name to `{name: SECRET_NAME, key: DATA_KEY}` |
| `objective.metric`, `direction` | Named finite final metric; Minimize (default) or Maximize |
| `parameters` | Map of names to search-space definitions |
| `trials`, `parallelism`, `seed` | Trial budget 1–100, parallelism 1..trials, unsigned 32-bit Random seed |
| `resources.cpu`, `memory`, `gpus` | Per-node CPU/memory requests+limits and optional GPU count |
| `require_cuda_proof` | Require built-in CUDA evidence matching the objective; GPU allocation alone is insufficient |
| `shared_data` | Optional one-time initializer and read-only shared PVC |
| `timeout_seconds` | Initializer/watch deadline, 1..86400 seconds (default 600) |

Search definitions:

```json
{
  "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]},
  "batch_size": {"type": "uniform", "min": 16, "max": 128, "integer": true},
  "learning_rate": {"type": "log_uniform", "min": 0.0001, "max": 0.1}
}
```

Names must be environment-compatible identifiers because assignments are injected into `KUBEFLOW_TRAINER_OPT_<name>`. Categories must be nonempty unique strings of at most 64 bytes. Floating ranges must have finite ordered bounds; integer ranges require integral endpoints. Log-uniform minima must be positive, and floating minima must be at least `1e-10` for the pinned Optuna provider. Only Random is implemented by the pinned controller.

User-supplied run names must be DNS labels of at most 28 characters to leave room for upstream Trial/JobSet/Pod suffixes. Omit `--name` to generate one. The namespace must already exist and the referenced runtime must meet its workload's requirements. Numeric resource syntax is checked locally; API admission and quotas remain authoritative.

Shared data example:

```json
{
  "shared_data": {
    "storage": "1Gi",
    "mount_path": "/dataset",
    "initializer": {
      "image": "your-registry/dataset-loader:tag",
      "command": ["download-dataset"],
      "args": ["--output", "/dataset/training.csv"]
    }
  }
}
```

`storage_class` is optional; omission uses the cluster default. RubixKube creates `<run>-data` (ReadWriteOnce PVC), `<run>-init` (one Job) and `<run>-runtime` (derived runtime). It does not modify the base runtime. The initializer is non-root with an fsGroup and a writable `/tmp`; choose a compatible image. All trial mounts are read-only. Existing shared resources are reused only when their ownership and recorded configuration match. A failed initializer is not silently retried; inspect its logs, then cancel/clean up or choose a new run name. An interrupted watch of a still-running initializer can be resumed by submitting the same name/config before an OptimizationJob exists.

ReadWriteOnce is suitable for this single-node local configuration. Choose a different storage/runtime design for multiple nodes. PVCs are deliberately retained after normal cancellation; explicit `--purge-shared-data` deletes them. Runtime clones are cluster-scoped when the base runtime is cluster-scoped, so run-name collisions across namespaces are rejected and cleanup verifies recorded namespace/configuration.
