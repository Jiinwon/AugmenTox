#!/bin/bash
#SBATCH --job-name=GNN_pipeline
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e

# ──────────────────────────────────────────────────────────────
# 1) 날짜/시간별 로그 디렉토리 생성
DATE=$(date +"%Y%m%d")      # ex) 20250512
TIME=$(date +"%H%M")        # ex) 1534
BASE_LOG="/home1/won0316/_Capstone/1_Git/Capstone_uos/unit_model/log"
LOG_DIR="${BASE_LOG}/${DATE}/${TIME}"
mkdir -p "${LOG_DIR}"

# 2) 이후 모든 stdout은 run.out에, stderr는 run.err에 기록
exec > "${LOG_DIR}/run.out" 2> "${LOG_DIR}/run.err"
# ──────────────────────────────────────────────────────────────

# self-nohup: 최초 실행 시 백그라운드 재실행
if [ -z "$NOHUP_MODE" ]; then
  echo "[*] Relaunching via nohup, logs → ${LOG_DIR}"
  NOHUP_MODE=1 nohup sbatch "$0" & 
  exit 0
fi

# 0. PYTHONPATH 설정 (프로젝트 루트)
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# (선택) 가상환경 활성화
# source /home1/won0316/anaconda3/envs/toxcast_env/bin/activate

# 1. Excel → CSV 변환
echo "[1/4] Converting Excel to CSV..."
python -u -m scripts.convert_excel_to_csv

# 2. Pre-training (ERα)
echo "[2/4] Pre-training on ERα..."
python -u -m train.pretrain

# 3. Fine-tuning (ERβ)
echo "[3/4] Fine-tuning on ERβ..."
python -u -m train.finetune

# 4. Embedding 시각화
echo "[4/4] Visualizing embeddings..."
python -u - << 'PYCODE'
import config.config as cfg
from data.load_data import load_data
from train.utils import load_model
from eval.visualize import visualize_embeddings
from models import gin, gcn, gat

device = cfg.DEVIDE

# 모델 클래스 선택
ModelClass = {"GIN": gin.GINNet, "GCN": gcn.GCNNet, "GAT": gat.GATNet}[cfg.MODEL_TYPE]

# 전체 테스트 세트로부터 input_dim 구하기
_, _, all_test = load_data(
    cfg.TARGET_DATA_PATH,
    train_ratio=0, val_ratio=0, test_ratio=1,
    random_seed=cfg.RANDOM_SEED
)
input_dim = all_test[0].x.shape[1]

# 모델 초기화 및 가중치 로드
model = ModelClass(
    input_dim=input_dim,
    hidden_dim=cfg.HIDDEN_DIM,
    output_dim=cfg.NUM_CLASSES,
    num_layers=cfg.NUM_LAYERS,
    dropout=cfg.DROPOUT,
    **({"heads": cfg.NUM_HEADS} if cfg.MODEL_TYPE=="GAT" else {})
)
load_model(model, cfg.FINETUNED_MODEL_PATH, device=device)

# 실제 테스트 데이터 준비
_, _, test_data = load_data(
    cfg.TARGET_DATA_PATH,
    train_ratio=cfg.TRAIN_RATIO,
    val_ratio=cfg.VAL_RATIO,
    test_ratio=cfg.TEST_RATIO,
    random_seed=cfg.RANDOM_SEED
)

# 시각화 및 저장
visualize_embeddings(model, test_data, device, save_path="embeddings.png")
print("✅ Saved embeddings.png")
PYCODE

echo "✅ All 4 steps completed successfully!"