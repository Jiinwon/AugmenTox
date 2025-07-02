#!/usr/bin/env bash
# runs_statistics.sh
# 모든 SOURCE_NAMES, TARGET_NAMES 조합에 대해
# 파인튜닝 모델과 타겟 전용 모델의 성능을 비교한다.

set -e

export MKL_THREADING_LAYER=GNU
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/.."

readarray -t SOURCE_NAMES < <(python3 - <<'PY'
import config.config as cfg
print("\n".join(cfg.SOURCE_NAMES))
PY
)
readarray -t TARGET_NAMES < <(python3 - <<'PY'
import config.config as cfg
print("\n".join(cfg.TARGET_NAMES))
PY
)
MODEL_TYPE=$(python3 - <<'PY'
import config.config as cfg
print(cfg.MODEL_TYPE)
PY
)

for SRC in "${SOURCE_NAMES[@]}"; do
  for TGT in "${TARGET_NAMES[@]}"; do
    export SOURCE_NAME="$SRC"
    export TARGET_NAME="$TGT"
    export MODEL_TYPE="$MODEL_TYPE"

    DATA_PATH="$(python3 - <<'PY'
import os, config.config as cfg
print(os.path.join(cfg.DATA_BASE_PATH, f"{os.environ['TARGET_NAME']}.csv"))
PY
)"

    RES_DIR="${SCRIPT_DIR}/results/${SRC}&&${TGT}_${MODEL_TYPE}"
    mkdir -p "$RES_DIR"

    FT_MODEL="$(python3 - <<'PY'
import config.config as cfg, os
cfg.SOURCE_NAME=os.environ['SOURCE_NAME']
cfg.TARGET_NAME=os.environ['TARGET_NAME']
cfg.MODEL_TYPE=os.environ['MODEL_TYPE']
print(cfg.FINETUNED_MODEL_PATH)
PY
)"

    TO_MODEL="$(python3 - <<'PY'
import config.config as cfg, os
cfg.TARGET_NAME=os.environ['TARGET_NAME']
cfg.MODEL_TYPE=os.environ['MODEL_TYPE']
print(cfg.TARGET_ONLY_MODEL_PATH)
PY
)"

    echo "[${SRC} -> ${TGT}] Repeated CV (finetuned)"
    python "$SCRIPT_DIR/run_repeated_cv.py" \
      --data-path "$DATA_PATH" \
      --val-idx-dir "$RES_DIR/ft_idx" \
      --model-path "$FT_MODEL" \
      --out-csv "$RES_DIR/ft_scores.csv"

    python "$SCRIPT_DIR/aggregate_results.py" \
      --input-csv "$RES_DIR/ft_scores.csv" \
      --out-json "$RES_DIR/ft_agg.json"

    echo "[${SRC} -> ${TGT}] Repeated CV (target-only)"
    python "$SCRIPT_DIR/run_repeated_cv.py" \
      --data-path "$DATA_PATH" \
      --val-idx-dir "$RES_DIR/to_idx" \
      --model-path "$TO_MODEL" \
      --out-csv "$RES_DIR/to_scores.csv"

    python "$SCRIPT_DIR/aggregate_results.py" \
      --input-csv "$RES_DIR/to_scores.csv" \
      --out-json "$RES_DIR/to_agg.json"

    python "$SCRIPT_DIR/start_test.py" \
      --json1 "$RES_DIR/ft_agg.json" \
      --json2 "$RES_DIR/to_agg.json" \
      --out-json "$RES_DIR/stat_test.json"

    python "$SCRIPT_DIR/visualize_results.py" \
      --json1 "$RES_DIR/ft_agg.json" \
      --json2 "$RES_DIR/to_agg.json" \
      --labels Finetuned TargetOnly \
      --out-dir "$RES_DIR"
  done
done

echo "✅ 모든 통계 검정 완료"

