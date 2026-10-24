# OptimizationJob development baseline

This repository holds reproducible local development tooling and pinned upstream revisions for the Kubeflow OptimizationJob effort. Product changes belong in the corresponding upstream checkouts, not in a replacement operator here.

The planning package is outside this repository at `H:\GSocProjects\optimizationjob-plan` and is not tracked. Evidence and session records are outside it at `H:\GSocProjects\optimizationjob-evidence`. The upstream Git checkouts are siblings under `H:\GSocProjects\kubeflow`.

## Real Kubernetes demo

The [full demo guide](demo/README.md) explains the orchestration logic, measured results, source code, inspection commands and cleanup. Run its complete CPU workflow from PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- bash /mnt/h/GSocProjects/optimizationjob/demo/start.sh
```

It uses the upstream OptimizationJob controller, Optuna, Trainer/JobSet, real linear-regression training and SDK metric push. The report independently checks the controller's winner against measured losses and worker delivery logs. Shared initialization, GPU execution and a new SDK submission backend remain outside this demo. See the guide for prerequisites and the local compatibility settings.

## Reproduce the Go baseline

From PowerShell:

```powershell
wsl -d Ubuntu-22.04 -- bash /mnt/h/GSocProjects/optimizationjob/scripts/smoke-trainer.sh
```

From Linux/WSL:

```bash
bash /mnt/h/GSocProjects/optimizationjob/scripts/smoke-trainer.sh
```

The script requires Bash, Python 3, Git and Go compatible with Trainer's Go 1.27 directive. It defaults to `/usr/local/go/bin/go`; set `GO_BINARY` to another compatible installation if needed. It verifies the clean Trainer checkout against `upstream.lock.json`, runs its actual utility and controller package tests, and saves full JSON event logs outside this repository. First execution may download Go modules. Existing logs with the `smoke-` prefix are replaced when rerun.

The script deliberately measures the pinned clean baseline. For later development commits or uncommitted upstream changes, use the relevant repository's documented test commands directly; do not rewrite the lock merely to bypass a mismatch.

Run the smoke runner's guard regression tests in WSL/Linux with Python's standard library:

```bash
cd /mnt/h/GSocProjects/optimizationjob
python3 -m unittest discover -s tests -p '*_test.py' -v
```

These use real temporary Git repositories to verify that dirty checkouts, corrupt indexes and revision mismatches cannot proceed to Go execution.

## Observed starting point at the first baseline

- Trainer already includes the typed `trainer.kubeflow.org/v1alpha1` OptimizationJob CRD, controller and generated clients. It currently orchestrates Random search; the schema exposes Grid but the controller rejects it.
- The SDK's existing optimizer still creates Katib Experiments. Its public `OptimizationJob` naming is not proof it submits the Trainer resource.
- Status push plumbing exists through `kubeflow.trainer.update_trainjob_status`; its current boolean/throttling behavior needs reconciliation with objective reporting guarantees.
- Shared model/dataset artifact publication across optimization trials is not established by this baseline.
- All three source revisions were recorded. The later demo verifies their narrow Random-search CPU integration; it does not prove provider restart recovery, all API validation cases or the remaining planned features.

## Original baseline environment

Windows and Ubuntu-22.04 on WSL2 can see the RTX 5060 Ti (16,311 MiB). WSL has Go 1.27.3 and a working Docker 29.1.3 daemon; Windows Docker Desktop's daemon was unavailable. Neither environment had a Kubernetes context at the first baseline. The later demo adds a dedicated kind cluster/private kubeconfig. WSL had neither Make nor GCC on PATH, Python 3.10 had no pip/ensurepip, and uv was not installed. Docker's registered runtimes did not include NVIDIA. The demo uses CPU; no GPU tensor execution is claimed.

An ignored Windows Python 3.13 venv now holds editable installations of the three pinned local SDK/API sources and minimal pytest dependencies. Its existing optimizer unit suite passes (50 tests). Reproduce from PowerShell, after restoring that environment from the external evidence's installation/dependency records:

```powershell
Set-Location H:\GSocProjects\kubeflow\sdk
& H:\GSocProjects\optimizationjob\.venv\Scripts\python.exe -m pytest kubeflow/optimizer -q
```

This verifies current Experiment-backed SDK behavior, not a new OptimizationJob backend or end-to-end deployment. No global Python packages were installed.

Git for Windows reports an ownership warning on H:'s filesystem. Use WSL Git for this workspace, or a command-scoped exception for this exact repository:

```powershell
git -c safe.directory=H:/GSocProjects/optimizationjob -C H:/GSocProjects/optimizationjob status
```

Do not disable ownership checks globally. No Git global configuration was changed.
