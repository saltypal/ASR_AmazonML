#!/usr/bin/env bash
set -euo pipefail

: "${BER_BUCKET:?Set BER_BUCKET to the private project bucket name.}"
BER_REGION="${BER_REGION:-ap-south-1}"
BER_REPO_URL="${BER_REPO_URL:-https://github.com/saltypal/ASR_AmazonML.git}"
BER_GIT_REF="${BER_GIT_REF:-main}"
BER_RUN_ID="${BER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BER_ROOT="${BER_ROOT:-/opt/amazonml}"
MODEL_REVISION="fd1525a9fd15316a2d503bf26ab031a61d056e98"

BER_ROOT="$(realpath -m "$BER_ROOT")"
case "$BER_ROOT" in
  /|/opt|/home|/root|/tmp)
    printf 'Refusing unsafe BER_ROOT: %s\n' "$BER_ROOT" >&2
    exit 2
    ;;
esac

sudo mkdir -p "$BER_ROOT"
sudo chown "$(id -u):$(id -g)" "$BER_ROOT"
cd "$BER_ROOT"

rm -rf "$BER_ROOT/source" "$BER_ROOT/.venv"
git clone --filter=blob:none "$BER_REPO_URL" source
git -C source checkout "$BER_GIT_REF"
GIT_COMMIT="$(git -C source rev-parse HEAD)"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e source/code/business_entity_resolution

rm -rf "$BER_ROOT/dataset" "$BER_ROOT/work" "$BER_ROOT/output" "$BER_ROOT/model-cache"
mkdir -p dataset work output model-cache
aws s3 sync "s3://${BER_BUCKET}/dataset/" dataset/ --region "$BER_REGION" --only-show-errors

python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "intfloat/multilingual-e5-small",
    revision="${MODEL_REVISION}",
    local_dir="${BER_ROOT}/model-cache/multilingual-e5-small",
)
PY

python - <<PY
from pathlib import Path
import yaml
base = Path("source/code/business_entity_resolution/configs/aws_g5.yaml")
config = yaml.safe_load(base.read_text(encoding="utf-8"))
config["embeddings"]["model_path"] = "${BER_ROOT}/model-cache/multilingual-e5-small"
Path("runtime_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

export PYTHONPATH="$BER_ROOT/source/code/business_entity_resolution/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see CUDA"; print(torch.__version__, torch.cuda.get_device_name(0))'

set +e
python -m ber.cli run-all \
  --config "$BER_ROOT/runtime_config.yaml" \
  --repo-root "$BER_ROOT/source" \
  --data-root "$BER_ROOT/dataset" \
  --work-dir "$BER_ROOT/work" \
  --output-dir "$BER_ROOT/output" 2>&1 | tee "$BER_ROOT/run.log"
PIPELINE_STATUS=${PIPESTATUS[0]}
set -e

DESTINATION="s3://${BER_BUCKET}/runs/${BER_RUN_ID}"
aws s3 cp "$BER_ROOT/run.log" "$DESTINATION/run.log" --region "$BER_REGION" --only-show-errors
if [[ -d "$BER_ROOT/output" ]]; then
  aws s3 sync "$BER_ROOT/output/" "$DESTINATION/output/" --region "$BER_REGION" --only-show-errors
fi
if [[ -d "$BER_ROOT/work/model" ]]; then
  aws s3 sync "$BER_ROOT/work/model/" "$DESTINATION/model/" --region "$BER_REGION" --only-show-errors
fi
for evidence in run_metadata.json completed_run.json; do
  if [[ -f "$BER_ROOT/work/$evidence" ]]; then
    aws s3 cp "$BER_ROOT/work/$evidence" "$DESTINATION/$evidence" --region "$BER_REGION" --only-show-errors
  fi
done

printf 'Run ID: %s\nGit commit: %s\nS3 destination: %s\n' "$BER_RUN_ID" "$GIT_COMMIT" "$DESTINATION"
exit "$PIPELINE_STATUS"
