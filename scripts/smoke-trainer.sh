#!/usr/bin/env bash
# Run the real upstream baseline tests. No cluster or SDK environment required.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
trainer_root=${1:-"$repo_root/../kubeflow/trainer"}
evidence_root=${2:-"$repo_root/../optimizationjob-evidence/runs/baseline"}
go_binary=${GO_BINARY:-/usr/local/go/bin/go}

if [[ ! -x "$go_binary" ]]; then
  go_binary=$(command -v go) || { echo "Go is missing; set GO_BINARY." >&2; exit 2; }
fi

expected_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["trainer"]["commit"])' "$repo_root/upstream.lock.json")
actual_sha=$(git -C "$trainer_root" rev-parse HEAD)
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Trainer revision differs from upstream.lock.json; review the new baseline first." >&2
  exit 2
fi
if [[ -n "$(git -C "$trainer_root" status --porcelain)" ]]; then
  echo "Trainer has local changes; use a clean checkout for a baseline measurement." >&2
  exit 2
fi

mkdir -p -- "$evidence_root"
evidence_root=$(cd -- "$evidence_root" && pwd)
cd -- "$trainer_root"
"$go_binary" version
for package in optimizationjob controller; do
  test_path=./pkg/controller
  if [[ "$package" == optimizationjob ]]; then
    test_path=./pkg/util/optimizationjob
  fi
  log_path="$evidence_root/smoke-$package.jsonl"
  echo "Testing $test_path at $actual_sha"
  CGO_ENABLED=0 "$go_binary" test -json "$test_path" -count=1 > "$log_path" 2>&1 || {
    tail -n 30 -- "$log_path"
    exit 1
  }
  python3 -c 'import json,sys; events=[json.loads(line) for line in open(sys.argv[1]) if line.startswith("{")]; passed=[e for e in events if e.get("Action")=="pass" and e.get("Test")]; failed=[e for e in events if e.get("Action")=="fail"]; print(f"{len(passed)} passing test/subtest events; {len(failed)} failures; log: {sys.argv[1]}"); sys.exit(0 if passed and not failed else 1)' "$log_path"
done
