#!/bin/bash

set -e

# 로그 디렉토리 설정
mkdir -p "$LOG_SUBDIR"
LOG_OUT="$LOG_SUBDIR/${SOURCE_NAME}&&${TARGET_NAME}_${MODEL_TYPE}.out"
LOG_ERR="$LOG_SUBDIR/${SOURCE_NAME}&&${TARGET_NAME}_${MODEL_TYPE}.err"
exec > "$LOG_OUT" 2> "$LOG_ERR"

# PYTHONPATH 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/.."

# 데이터 경로
DATA_PATH=$(python3 - <<'PY'
import os, config.config as cfg
print(os.path.join(cfg.DATA_BASE_PATH, f"{os.environ['TARGET_NAME']}.csv"))
PY
)

# 결과 디렉토리
RES_DIR="${SCRIPT_DIR}/results/${SOURCE_NAME}&&${TARGET_NAME}_${MODEL_TYPE}"
mkdir -p "$RES_DIR"

# 모델 경로들
FT_MODEL=$(python3 - <<'PY'
import os, config.config as cfg
cfg.SOURCE_NAME=os.environ['SOURCE_NAME']
cfg.TARGET_NAME=os.environ['TARGET_NAME']
cfg.MODEL_TYPE=os.environ['MODEL_TYPE']
print(cfg.FINETUNED_MODEL_PATH)
PY
)
TO_MODEL=$(python3 - <<'PY'
import os, config.config as cfg
cfg.TARGET_NAME=os.environ['TARGET_NAME']
cfg.MODEL_TYPE=os.environ['MODEL_TYPE']
print(cfg.TARGET_ONLY_MODEL_PATH)
PY
)

echo "[${SOURCE_NAME} -> ${TARGET_NAME}] Repeated CV (finetuned)"
python "$SCRIPT_DIR/run_repeated_cv.py" \
  --data-path "$DATA_PATH" \
  --val-idx-dir "$RES_DIR/ft_idx" \
  --model-path "$FT_MODEL" \
  --out-csv "$RES_DIR/ft_scores.csv"

python "$SCRIPT_DIR/aggregate_results.py" \
  --input-csv "$RES_DIR/ft_scores.csv" \
  --out-json "$RES_DIR/ft_agg.json"

echo "[${SOURCE_NAME} -> ${TARGET_NAME}] Repeated CV (target-only)"
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

echo "✅ 통계 검정 완료: ${SOURCE_NAME} -> ${TARGET_NAME} [${MODEL_TYPE}]"
