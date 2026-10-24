"""Check the demonstration trains a model rather than inventing trial scores."""

import math
import unittest

from training import train


class TrainingTest(unittest.TestCase):
    def test_larger_stable_learning_rate_learns_the_linear_relationship(self) -> None:
        slow = train(0.005, epochs=40)
        fast = train(0.1, epochs=40)
        self.assertLess(fast["val_loss"], slow["val_loss"])
        self.assertLess(fast["val_loss"], 0.02)
        self.assertAlmostEqual(fast["weight"], 2.0, delta=0.2)
        self.assertAlmostEqual(fast["bias"], 1.0, delta=0.1)
        self.assertTrue(all(math.isfinite(row["val_loss"]) for row in fast["history"]))

    def test_same_configuration_is_reproducible(self) -> None:
        self.assertEqual(train(0.02, epochs=10), train(0.02, epochs=10))

    def test_nonfinite_or_unstable_configuration_is_rejected(self) -> None:
        for rate in [0.0, -0.1, math.nan, math.inf, 10.0]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                train(rate)
        with self.assertRaises(ValueError):
            train(0.1, epochs=0)


if __name__ == "__main__":
    unittest.main()
