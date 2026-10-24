#!/usr/bin/env bash
# Full local CPU demo; intentionally uses its own kind cluster and kubeconfig.
set -euo pipefail
demo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$demo/.." && pwd)
workspace=$(cd -- "$repo/.." && pwd)
state="$repo/.tools/demo"
evidence="$workspace/optimizationjob-evidence/runs/demo"
go_binary=${GO_BINARY:-/usr/local/go/bin/go}
mkdir -p -- "$state/build/manager-image" "$state/build/training" "$evidence"

for tool in git python3 docker kind kubectl; do
  command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 2; }
done
[[ -x "$go_binary" ]] || { echo "Set GO_BINARY to a Go 1.27-compatible executable" >&2; exit 2; }
python3 - "$repo" <<'PY'
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
for name, item in json.loads((root / "upstream.lock.json").read_text())["repositories"].items():
    checkout = (root / item["checkout"]).resolve()
    sha = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    # These checkouts were created by Git for Windows with autocrlf=true.
    # Match that interpretation per command; do not rewrite files or global config.
    status = subprocess.check_output(["git", "-c", "core.autocrlf=true", "-C", str(checkout),
                                      "status", "--porcelain"], text=True)
    if sha != item["commit"] or status:
        raise SystemExit(f"{name} must be clean and match upstream.lock.json; inspect before deploying")
PY

clusters=$(kind get clusters)
if ! grep -Fxq optimizationjob-demo <<< "$clusters"; then
  kind create cluster --name optimizationjob-demo --config "$demo/kind.yaml" \
    --kubeconfig "$state/kubeconfig" --wait 120s
fi
k=(kubectl --kubeconfig "$state/kubeconfig" --context kind-optimizationjob-demo)
"${k[@]}" wait --for=condition=Ready nodes --all --timeout=120s

(cd -- "$workspace/kubeflow/trainer" && CGO_ENABLED=0 "$go_binary" build \
  -o "$state/build/manager-image/manager" ./cmd/trainer-controller-manager)
cp -- "$demo/Dockerfile.manager" "$state/build/manager-image/Dockerfile"
docker build --platform linux/amd64 --provenance=false \
  -t optimizationjob-demo-manager:b12f968 "$state/build/manager-image"

training="$state/build/training"
mkdir -p -- "$training/trainer-api" "$training/katib-api" "$training/sdk"
cp -a -- "$workspace/kubeflow/trainer/api/python_api/." "$training/trainer-api/"
cp -a -- "$workspace/kubeflow/katib/api/python_api/." "$training/katib-api/"
for item in pyproject.toml README.md LICENSE kubeflow; do
  cp -a -- "$workspace/kubeflow/sdk/$item" "$training/sdk/"
done
cp -- "$demo/training.py" "$training/training.py"
cp -- "$demo/Dockerfile.training" "$training/Dockerfile"
docker build --platform linux/amd64 --provenance=false \
  -t optimizationjob-demo-training:local "$training"
provider_digest=ghcr.io/kubeflow/katib/suggestion-optuna@sha256:04a71f288d04251835846b87d18a9fcf816eb59003d282555dc3d94412c879c7
docker pull "$provider_digest"
docker tag "$provider_digest" ghcr.io/kubeflow/katib/suggestion-optuna:v0.19.0
docker image save --platform linux/amd64 -o "$state/images-amd64.tar" \
  optimizationjob-demo-manager:b12f968 optimizationjob-demo-training:local \
  ghcr.io/kubeflow/katib/suggestion-optuna:v0.19.0
kind load image-archive --name optimizationjob-demo "$state/images-amd64.tar"

"${k[@]}" apply --server-side -k "$demo/deploy"
"${k[@]}" rollout status deployment/jobset-controller-manager -n kubeflow-system --timeout=300s
"${k[@]}" rollout status deployment/kubeflow-trainer-controller-manager -n kubeflow-system --timeout=300s
"${k[@]}" apply --server-side -f "$demo/runtime.yaml"
python3 "$demo/run.py" --output "$evidence/reports" "$@"
echo "Open $evidence/reports/index.html for the walkthrough and observed results."
