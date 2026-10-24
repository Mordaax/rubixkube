"""Small CPU-only linear regression workload with real held-out loss."""

import json
import math
import os
import random
import time
from typing import Any


def train(learning_rate: float, epochs: int = 40) -> dict[str, Any]:
    """Fit y = weight*x + bias using full-batch gradient descent."""
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.2 or epochs < 1:
        raise ValueError("Use a finite learning rate in (0, 0.2] and positive epochs")
    rng = random.Random(17)
    examples = []
    for _ in range(128):
        x = rng.uniform(-1.0, 1.0)
        examples.append((x, 2.0 * x + 1.0 + rng.gauss(0.0, 0.03)))
    training, validation = examples[:64], examples[64:]
    weight, bias = 0.0, 0.0
    history = []
    for epoch in range(1, epochs + 1):
        errors = [(weight * x + bias - y, x) for x, y in training]
        weight -= learning_rate * 2.0 * sum(error * x for error, x in errors) / len(training)
        bias -= learning_rate * 2.0 * sum(error for error, _ in errors) / len(training)
        loss = sum((weight * x + bias - y) ** 2 for x, y in validation) / len(validation)
        history.append({"epoch": epoch, "val_loss": loss})
    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "weight": weight,
        "bias": bias,
        "val_loss": history[-1]["val_loss"],
        "history": history,
        "execution": "CPU, Python standard library; no GPU or simulated score",
    }


def main() -> None:
    from kubeflow.trainer import update_trainjob_status

    result = train(float(os.environ["KUBEFLOW_TRAINER_OPT_learning_rate"]))
    print(json.dumps({"event": "training_complete", **result}), flush=True)
    # Report only at the end to avoid dropping the final objective in the SDK's throttle.
    for attempt in range(8):
        if update_trainjob_status(progress_percent=100, metrics={"val_loss": result["val_loss"]}):
            print(json.dumps({"event": "metric_delivered", "attempt": attempt + 1}), flush=True)
            return
        time.sleep(5.5)
    raise RuntimeError("Objective metric was not acknowledged by the Trainer status server")


if __name__ == "__main__":
    main()
