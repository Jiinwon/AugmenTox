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

# 각 SOURCE_NAME과 TARGET_NAME의 모든 조합에 대해 Slurm job 제출
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
           --export=ALL,SOURCE_NAME="${SRC}",TARGET_NAME="${TGT}",MODEL_TYPE="${MODEL_TYPE}",LOG_SUBDIR="${BASE_LOG}" \
           "${SCRIPT_DIR}/run_single_pipeline.sh"
  done
done