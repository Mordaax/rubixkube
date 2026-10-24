# Changelog

## 0.2.0

- Configurable typed OptimizationJob submission for user images, arbitrary single objectives, multiple parameters, Random categorical/uniform/log-uniform spaces, budgets and concurrency.
- Explicit-context doctor, status, list, watch/resume, logs, results and ownership-checked cancellation; shared PVC retention with explicit purge.
- One-time initializer Job and PVC, with a derived runtime mounting shared artifacts read-only into trials.
- CSV input and variable epochs for CPU/CUDA built-in training; actual dynamic CUDA execution verified against CPU. User models run through their own images.
- Generic authenticated metric-reporting API/command, Secret references, static signed-objective plots and atomic result snapshots including Pod evidence.
- Installable rubixkube binary, compatibility executable, completions, packaged worker Dockerfile, license notices, configuration/workload/quickstart docs and CI configuration.
- Local acceptance: shared 256-row dataset, custom objective and two parameters on actual GPU; CPU continuous/integer search; resume/log/status; cancellation retains data and explicit later purge succeeds.

This local single-node release is not a production certification or replacement of the upstream Kubeflow controller/SDK implementations. CI GPU tests are hardware-independent unless explicitly run on a device.
