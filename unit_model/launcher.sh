#!/bin/bash

#SBATCH --job-name=GNN_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

# 현재 스크립트 경로
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"

# config.py에서 OPERA 읽기
OPERA=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.OPERA)
PYCODE
)

# 로그 디렉토리 생성
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H%M")
BASE_LOG="./log/${DATE}/${TIME}"
mkdir -p "${BASE_LOG}"

# config.py에서 SOURCE_NAMES, TARGET_NAMES, MODEL_TYPE 읽기
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

# 길이 체크
if [ "${#SOURCE_NAMES[@]}" -ne "${#TARGET_NAMES[@]}" ]; then
  echo "SOURCE_NAMES와 TARGET_NAMES의 길이가 다릅니다." >&2
  exit 1
fi

# OPERA가 True인 경우 SDF 파일을 사용하여 단일 SOURCE_NAME과 모든 TARGET_NAME 조합 수행
if [ "$OPERA" = "True" ]; then
    echo "OPERA 데이터로 pretraining을 수행합니다."
    # config에서 SDF 경로 읽기
    SDF_PATH=$(python3 - << 'PYCODE'
import config.config as cfg
print(cfg.SOURCE_SDF_PATH)
PYCODE
    )
    # SDF 파일명을 SRC 변수로 사용 (확장자 제거)
    SRC=$(basename "$SDF_PATH" .sdf)

    # 단일 SRC에 대해 모든 TGT 반복
    for TGT in "${TARGET_NAMES[@]}"; do
        PAIR="${SRC}&&${TGT}_${MODEL_TYPE}"
        OUT="${BASE_LOG}/${PAIR}.out"
        ERR="${BASE_LOG}/${PAIR}.err"
        sbatch --job-name="GNN_${SRC}_${TGT}" \
               --partition=gpu1 \
               --gres=gpu:rtx3090:1 \
               --cpus-per-task=8 \
               --output="${OUT}" \
               --error="${ERR}" \
               --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="$MODEL_TYPE",LOG_SUBDIR="${BASE_LOG}",OPERA="${OPERA}" \
               "${SCRIPT_DIR}/run_single_pipeline.sh"
    done
else
    # 기존 방식: SOURCE_NAMES와 TARGET_NAMES의 모든 조합 수행
    for SRC in "${SOURCE_NAMES[@]}"; do
        for TGT in "${TARGET_NAMES[@]}"; do
            PAIR="${SRC}&&${TGT}_${MODEL_TYPE}"
            OUT="${BASE_LOG}/${PAIR}.out"
            ERR="${BASE_LOG}/${PAIR}.err"
            sbatch --job-name="GNN_${SRC}_${TGT}" \
                   --partition=gpu1 \
                   --gres=gpu:rtx3090:1 \
                   --cpus-per-task=8 \
                   --output="${OUT}" \
                   --error="${ERR}" \
                   --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="${MODEL_TYPE}",LOG_SUBDIR="${BASE_LOG}",OPERA="${OPERA}" \
                   "${SCRIPT_DIR}/run_single_pipeline.sh"
        done
    done
fi