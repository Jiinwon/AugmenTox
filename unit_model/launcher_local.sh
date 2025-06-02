#!/bin/bash

set -e

# Conda 환경 활성화 (필요한 경우)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ubai

# 현재 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"

# 로그 디렉토리 생성
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H%M")
BASE_LOG="${SCRIPT_DIR}/log/${DATE}/${TIME}"
mkdir -p "$BASE_LOG"

# config.py에서 SOURCE_NAMES, TARGET_NAMES, MODEL_TYPE, OPERA 불러오기
readarray -t SOURCE_NAMES < <(python3 - << 'PYCODE'
import config.config as cfg
for n in cfg.SOURCE_NAMES: print(n)
PYCODE
)

readarray -t TARGET_NAMES < <(python3 - << 'PYCODE'
import config.config as cfg
for n in cfg.TARGET_NAMES: print(n)
PYCODE
)

MODEL_TYPE=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.MODEL_TYPE)
PYCODE
)

OPERA=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.OPERA)
PYCODE
)

# 길이 체크
if [ "$OPERA" = "False" ] && [ "${#SOURCE_NAMES[@]}" -ne "${#TARGET_NAMES[@]}" ]; then
  echo "SOURCE_NAMES와 TARGET_NAMES의 길이가 다릅니다." >&2
  exit 1
fi

# OPERA = True: 단일 SDF source에 대해 모든 TARGET 실행
if [ "$OPERA" = "True" ]; then
    SDF_PATH=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.SOURCE_SDF_PATH)
PYCODE
    )
    SRC=$(basename "$SDF_PATH" .sdf)

    for TGT in "${TARGET_NAMES[@]}"; do
        echo "🔹 실행: $SRC -> $TGT ($MODEL_TYPE)"
        SOURCE_NAME="$SRC" TARGET_NAME="$TGT" MODEL_TYPE="$MODEL_TYPE" OPERA="True" LOG_SUBDIR="$BASE_LOG" \
        bash "$SCRIPT_DIR/run_single_pipeline.sh"
    done
else
    # SOURCE_NAMES × TARGET_NAMES 모든 조합 실행 (단, 같은 이름은 제외)
    for SRC in "${SOURCE_NAMES[@]}"; do
        for TGT in "${TARGET_NAMES[@]}"; do
            if [ "$SRC" != "$TGT" ]; then
                echo "🔹 실행: $SRC -> $TGT ($MODEL_TYPE)"
                SOURCE_NAME="$SRC" TARGET_NAME="$TGT" MODEL_TYPE="$MODEL_TYPE" OPERA="False" LOG_SUBDIR="$BASE_LOG" \
                bash "$SCRIPT_DIR/run_single_pipeline.sh"
            else
                echo "⚠️ 건너뜀 (동일한 SOURCE/TARGET): $SRC"
            fi
        done
    done
fi

echo "✅ 모든 실험 실행 완료"
