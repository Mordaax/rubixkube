"""Submit or inspect a real OptimizationJob and produce a checked local report."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any


def summarize(job: dict[str, Any], trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Require successful measured trials and independently verify the selected winner."""
    if not any(c["type"] == "Complete" and c["status"] == "True"
               for c in job.get("status", {}).get("conditions", [])):
        raise ValueError("OptimizationJob has not completed")
    if len(trials) != job["spec"]["numTrials"]:
        raise ValueError("Observed trial count does not match the job budget")
    rows = []
    objective = job["spec"]["objectives"][0]
    for trial in trials:
        conditions = trial.get("status", {}).get("conditions", [])
        if not any(c["type"] == "Complete" and c["status"] == "True" for c in conditions):
            raise ValueError("A trial has not completed successfully")
        metrics = trial.get("status", {}).get("trainerStatus", {}).get("metrics", [])
        values = [float(m["value"]) for m in metrics if m["name"] == objective["metric"]]
        if not values or not math.isfinite(values[-1]):
            raise ValueError("A trial has no finite objective observation")
        env = {e["name"]: e.get("value", "") for e in trial["spec"]["trainer"]["env"]}
        rows.append({
            "name": trial["metadata"]["name"],
            "created": trial["metadata"].get("creationTimestamp", ""),
            "learning_rate": float(env["KUBEFLOW_TRAINER_OPT_learning_rate"]),
            "val_loss": values[-1],
            "status": "Complete",
        })
    direction = 1 if objective["direction"] == "Minimize" else -1
    rows.sort(key=lambda row: (direction * row["val_loss"], row["created"], row["name"]))
    selected = job["status"]["result"]["trainJobName"]
    if selected != rows[0]["name"]:
        raise ValueError("Controller winner differs from the measured objective ranking")
    return {"job": job["metadata"]["name"], "best": rows[0], "trials": rows,
            "objective": objective, "winner_verified": True}


def kubectl(kubeconfig: Path, *args: str, payload: dict[str, Any] | None = None) -> str:
    command = ["kubectl", "--kubeconfig", str(kubeconfig), "--context",
               "kind-optimizationjob-demo", *args]
    result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                            text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", help="Inspect a completed/existing demo job instead of submitting")
    parser.add_argument("--kubeconfig", type=Path,
                        default=here.parent / ".tools/demo/kubeconfig")
    parser.add_argument("--output", type=Path,
                        default=here.parent.parent / "optimizationjob-evidence/runs/demo/reports")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    name = args.existing
    if not name:
        manifest = json.loads(kubectl(args.kubeconfig, "create", "--dry-run=client", "-f",
                                      str(here / "optimizationjob.yaml"), "-o", "json"))
        name = "learning-rate-demo-" + datetime.now(timezone.utc).strftime("%m%d%H%M%S")
        manifest["metadata"]["name"] = name
        kubectl(args.kubeconfig, "create", "-f", "-", payload=manifest)
    print(f"Watching real OptimizationJob {name}", flush=True)
    deadline = time.monotonic() + args.timeout
    previous = None
    while True:
        job = json.loads(kubectl(args.kubeconfig, "get", "optimizationjob", name,
                                 "-n", "optimizationjob-demo", "-o", "json"))
        trials = json.loads(kubectl(args.kubeconfig, "get", "trainjobs", "-n",
                                    "optimizationjob-demo", "-l",
                                    f"trainer.kubeflow.org/optimization-job={name}",
                                    "-o", "json"))["items"]
        state = [(c["type"], c["status"]) for c in job.get("status", {}).get("conditions", [])]
        progress = (len(trials), str(state))
        if progress != previous:
            print(f"Observed {len(trials)} trials; conditions {state}", flush=True)
            previous = progress
        if ("Failed", "True") in state:
            (args.output / "failed-job.json").write_text(json.dumps(job, indent=2))
            raise RuntimeError(f"OptimizationJob failed: {job['status']['conditions']}")
        if ("Complete", "True") in state:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Real OptimizationJob did not complete before the deadline")
        time.sleep(2)
    report = summarize(job, trials)
    pods = json.loads(kubectl(args.kubeconfig, "get", "pods", "-n", "optimizationjob-demo",
                              "-o", "json"))["items"]
    for row in report["trials"]:
        trial_pods = [p for p in pods if p["metadata"]["name"].startswith(row["name"] + "-")]
        if len(trial_pods) != 1 or trial_pods[0].get("status", {}).get("phase") != "Succeeded":
            raise ValueError("Expected exactly one successful real training Pod per trial")
        pod = trial_pods[0]["metadata"]["name"]
        logs = kubectl(args.kubeconfig, "logs", pod, "-n", "optimizationjob-demo", "-c", "node")
        (args.output / f"{row['name']}.log").write_text(logs)
        events = [json.loads(line) for line in logs.splitlines() if line.startswith('{"event":')]
        trained = [e for e in events if e.get("event") == "training_complete"]
        delivered = [e for e in events if e.get("event") == "metric_delivered"]
        if len(trained) != 1 or not delivered:
            raise ValueError("Pod logs do not prove model training and acknowledged metric delivery")
        if not math.isclose(trained[0]["val_loss"], row["val_loss"], rel_tol=1e-12):
            raise ValueError("Reported objective differs from the model's measured loss")
        row.update({"pod": pod, "history": trained[0]["history"], "weight": trained[0]["weight"],
                    "bias": trained[0]["bias"], "metric_delivered": True})
    report.update({
        "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "mode": "Real Kubernetes CPU run",
        "cluster": "kind-optimizationjob-demo",
        "training": "64 training + 64 validation samples; 40 gradient descent epochs; y = 2x + 1 + noise",
        "coverage": {"real_crd": True, "real_trainjobs": True, "real_optuna": True,
                     "sdk_push_metrics": True, "new_sdk_submission_backend": False,
                     "shared_initializer": False, "gpu": False},
        "source_pins": json.loads((here.parent / "upstream.lock.json").read_text())["repositories"],
    })
    (args.output / "optimizationjob.json").write_text(json.dumps(job, indent=2))
    (args.output / "trainjobs.json").write_text(json.dumps(trials, indent=2))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    template = (here / "report-template.html").read_text()
    data = json.dumps(report).replace("<", "\\u003c")
    (args.output / "index.html").write_text(template.replace("__DEMO_DATA__", data))
    print(f"VERIFIED WINNER: {report['best']['name']}")
    print(f"learning_rate={report['best']['learning_rate']}; val_loss={report['best']['val_loss']:.8f}")
    print(f"Real results and walkthrough: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
