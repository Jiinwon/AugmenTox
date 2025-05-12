#!/usr/bin/env bash
set -e

# 로그 디렉토리 생성
LOG_DIR="/home1/won0316/_Capstone/1_Git/Capstone_uos/unit_model/log"
mkdir -p "${LOG_DIR}"

# 만약 NOHUP_MODE 환경변수가 설정되지 않았다면 nohup으로 self 실행
if [ -z "$NOHUP_MODE" ]; then
  echo "[*] Launching self via nohup..."
  NOHUP_MODE=1 nohup bash "$0" > "${LOG_DIR}/quick_experiment.log" 2>&1 &
  exit 0
fi

# PYTHONPATH 설정
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# 가상환경 활성화 (필요 시)
# source /home1/won0316/anaconda3/envs/toxcast_env/bin/activate

echo "[1/1] Quick experiment: 2 epochs each (pretrain & finetune)"

python -u << 'PYCODE'
import config.config as cfg
cfg.NUM_EPOCHS_PRETRAIN = 2
cfg.NUM_EPOCHS_FINETUNE = 2

from train.pretrain import run_pretraining
from train.finetune import run_finetuning

print(">>> Pretraining (2 epochs)...")
run_pretraining()
print(">>> Finetuning (2 epochs)...")
run_finetuning()
PYCODE

echo "✅ Quick experiment completed!"
