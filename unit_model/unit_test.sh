#!/bin/bash
#SBATCH --job-name=GNN_pipeline
#SBATCH --partition=gpu4
##SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# → Slurm 기본 출력 로그를 사용하지 않음
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

# ──────────────────────────────────────────────────────────────
# 1) 날짜/시간별 로그 디렉토리 생성
DATE=$(date +"%Y%m%d")      # 예) 20250512
TIME=$(date +"%H%M")        # 예) 1534
BASE_LOG="./log"
LOG_DIR="${BASE_LOG}/${DATE}/${TIME}"
mkdir -p "${LOG_DIR}"

# 2) 이후의 stdout은 run.out에, stderr는 run.err에 기록
exec > "${LOG_DIR}/run.out" 2> "${LOG_DIR}/run.err"
# ──────────────────────────────────────────────────────────────

# pretraining, finetuning data name
# 프로젝트 루트(이 스크립트가 있는 디렉토리)를 PYTHONPATH에 추가
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# config에 정의된 SOURCE_NAME, TARGET_NAME 읽어오기
SOURCE_NAME=$(python - << 'PYCODE'
import config.config as cfg
print(cfg.SOURCE_NAME)
PYCODE
)

TARGET_NAME=$(python - << 'PYCODE'
import config.config as cfg
print(cfg.TARGET_NAME)
PYCODE
)

# self-nohup: 처음 실행 시에만 백그라운드로 재실행
if [ -z "$NOHUP_MODE" ]; then
  echo "[*] nohup으로 재실행, 로그 → ${LOG_DIR}"
  NOHUP_MODE=1 nohup sbatch "$0" &
  exit 0
fi

# 3) PYTHONPATH를 프로젝트 루트로 설정
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# (필요시) 가상환경 활성화
# source /home1/won0316/anaconda3/envs/toxcast_env/bin/activate

# ──────────── 워크플로우 ────────────

# 1/4: Excel → CSV 변환 (이미 생성되어 있으면 스킵)
echo "[1/4] Converting Excel to CSV..."
if [ ! -f "$(python -c 'import config.config as cfg; print(cfg.SOURCE_DATA_PATH)')" ] || \
   [ ! -f "$(python -c 'import config.config as cfg; print(cfg.TARGET_DATA_PATH)')" ]; then
  python -u -m scripts.convert_excel_to_csv
else
  echo "→ 변환 스킵: SOURCE_DATA_PATH와 TARGET_DATA_PATH가 모두 존재합니다."
fi

# 2/4: ERα 사전학습 (Pre-training)
echo "[2/4] Pre-training on ${SOURCE_NAME}..."
python -u -m train.pretrain

# 3/4: ERβ 파인튜닝 (Fine-tuning)
echo "[3/4] Fine-tuning on ${TARGET_NAME}..."
python -u -m train.finetune

# 3.1/4: ERβ 전용 학습 (Target-only)
echo "[3.1/4] Target-only training on ${TARGET_NAME}..."
python -u -m train.target_only

# 4/4: Embedding 시각화
echo "[4/4] Visualizing embeddings..."
python -u - << 'PYCODE'
import torch
import config.config as cfg
from data.load_data import load_data
from train.utils import load_model
from eval.visualize import visualize_embeddings
from models import gin, gcn, gat

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 클래스 선택
ModelClass = {"GIN": gin.GINNet, "GCN": gcn.GCNNet, "GAT": gat.GATNet}[cfg.MODEL_TYPE]

# 전체 테스트 세트에서 input_dim 계산
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

# Embedding 시각화 및 저장
visualize_embeddings(model, test_data, device, save_path="./figure/embeddings.png")
print("✅ Saved embeddings.png")
PYCODE

echo "✅ 모든 단계가 성공적으로 완료되었습니다!"