"""Exercise the smoke runner's refusal paths with real temporary Git repositories."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SmokeTrainerGuardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="optimizationjob-smoke-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runner = self.root / "baseline" / "scripts" / "smoke-trainer.sh"
        self.runner.parent.mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/smoke-trainer.sh", self.runner)
        self.trainer = self.root / "trainer"
        subprocess.run(["git", "init", "-q", str(self.trainer)], check=True)
        subprocess.run(
            [
                "git", "-C", str(self.trainer), "-c", "user.name=Baseline Fixture",
                "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty",
                "-m", "fixture", "-q",
            ],
            check=True,
        )
        sha = subprocess.check_output(
            ["git", "-C", str(self.trainer), "rev-parse", "HEAD"], text=True
        ).strip()
        self.lock = self.runner.parent.parent / "upstream.lock.json"
        self.lock.write_text(json.dumps({"repositories": {"trainer": {"commit": sha}}}))
        self.marker = self.root / "go-was-run"
        self.go = self.root / "go"
        self.go.write_text('#!/bin/sh\n: > "$GO_MARKER"\nexit 23\n')
        self.go.chmod(0o755)

    def run_runner(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.runner), str(self.trainer), str(self.root / "evidence")],
            env={**os.environ, "GO_BINARY": str(self.go), "GO_MARKER": str(self.marker)},
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_unreadable_git_index_prevents_test_execution(self) -> None:
        (self.trainer / ".git" / "index").write_bytes(b"not a git index")
        result = self.run_runner()
        self.assertEqual(result.returncode, 128, result.stderr)
        self.assertFalse(self.marker.exists(), "An unverified checkout must not run Go")

    def test_dirty_checkout_prevents_test_execution(self) -> None:
        (self.trainer / "untracked.txt").write_text("local user work\n")
        result = self.run_runner()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.marker.exists())

    def test_revision_mismatch_prevents_test_execution(self) -> None:
        self.lock.write_text(json.dumps({"repositories": {"trainer": {"commit": "0" * 40}}}))
        result = self.run_runner()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
