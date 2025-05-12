#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────
# 로그 디렉토리 생성
LOG_DIR="/home1/won0316/_Capstone/1_Git/Capstone_uos/unit_model/log"
mkdir -p "${LOG_DIR}"

# 모든 stdout/stderr를 로그 파일로 리다이렉트
exec > "${LOG_DIR}/quick_experiment.log" 2>&1
# ──────────────────────────────────────────────────────────────

# 0. 프로젝트 루트(이 스크립트가 있는 디렉토리)를 PYTHONPATH에 추가
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# (선택) 가상환경 활성화
# source /home1/won0316/anaconda3/envs/toxcast_env/bin/activate

echo "[1/1] Quick experiment: 5 epochs each (pretrain & finetune)"

python -u << 'PYCODE'
import config.config as cfg

# 에폭 수 짧게 지정
cfg.NUM_EPOCHS_PRETRAIN = 5
cfg.NUM_EPOCHS_FINETUNE = 5

from train.pretrain import run_pretraining
from train.finetune import run_finetuning

print(">>> Pretraining (5 epochs)...")
run_pretraining()

print(">>> Finetuning (5 epochs)...")
run_finetuning()
PYCODE

echo "✅ Quick experiment completed!"
