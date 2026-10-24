"""A demo report must reject incomplete runs or an incorrect controller winner."""

import copy
import unittest

from run import summarize


class ReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.job = {
            "metadata": {"name": "example"},
            "spec": {"numTrials": 2, "objectives": [{"metric": "val_loss", "direction": "Minimize"}]},
            "status": {"conditions": [{"type": "Complete", "status": "True"}],
                       "result": {"trainJobName": "trial-b"}},
        }
        self.trials = []
        for name, rate, loss in [("trial-a", "0.005", "0.8"), ("trial-b", "0.1", "0.003")]:
            self.trials.append({
                "metadata": {"name": name, "creationTimestamp": "2026-10-24T10:00:00Z"},
                "spec": {"trainer": {"env": [{"name": "KUBEFLOW_TRAINER_OPT_learning_rate", "value": rate}]}},
                "status": {"conditions": [{"type": "Complete", "status": "True"}],
                           "trainerStatus": {"metrics": [{"name": "val_loss", "value": loss}]}},
            })

    def test_uses_measured_metrics_and_checks_the_controller_winner(self) -> None:
        result = summarize(self.job, self.trials)
        self.assertEqual(result["best"]["name"], "trial-b")
        self.assertEqual(result["best"]["val_loss"], 0.003)
        self.assertEqual(len(result["trials"]), 2)

    def test_wrong_winner_is_not_presented_as_success(self) -> None:
        self.job["status"]["result"]["trainJobName"] = "trial-a"
        with self.assertRaises(ValueError):
            summarize(self.job, self.trials)

    def test_partial_and_nonfinite_results_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize(self.job, self.trials[:1])
        invalid = copy.deepcopy(self.trials)
        invalid[0]["status"]["trainerStatus"]["metrics"][0]["value"] = "NaN"
        with self.assertRaises(ValueError):
            summarize(self.job, invalid)
        self.job["status"]["conditions"] = []
        with self.assertRaises(ValueError):
            summarize(self.job, self.trials)


if __name__ == "__main__":
    unittest.main()
