#!/usr/bin/env bash
set -e

# 로그 디렉토리 설정
LOG_DIR="/home1/won0316/_Capstone/1_Git/Capstone_uos/unit_model/log"
mkdir -p "${LOG_DIR}"

# 현재 날짜-시간 형식 지정 (예: 20250512_1430)
NOW=$(date +"%Y%m%d_%H%M")

# self-nohup 실행
if [ -z "$NOHUP_MODE" ]; then
  echo "[*] Launching unittest.sh via nohup..."
  NOHUP_MODE=1 nohup bash "$0" > "${LOG_DIR}/unittest_${NOW}.log" 2>&1 &
  exit 0
fi

# 0. 프로젝트 루트(이 스크립트가 있는 디렉토리)를 PYTHONPATH에 추가
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# (선택) 가상환경 활성화
# source /home1/won0316/anaconda3/envs/toxcast_env/bin/activate

# 1. Excel → CSV 변환
echo "[1/4] Converting Excel to CSV..."
python -m scripts.convert_excel_to_csv

# 2~4. Pre-training + Fine-tuning + 시각화 (에폭 2로 제한)
echo "[2/4] Pre-training on ERα..."
echo "[3/4] Fine-tuning on ERβ..."
echo "[4/4] Visualizing embeddings..."
python -u - << 'PYCODE'
import config.config as cfg
from data.load_data import load_data
from train import pretrain, finetune
from train.utils import load_model
from eval.visualize import visualize_embeddings

# 하이퍼파라미터 조정 (짧은 실험용)
cfg.NUM_EPOCHS_PRETRAIN = 2
cfg.NUM_EPOCHS_FINETUNE = 2

# 사전학습 및 파인튜닝
pretrain.run_pretraining()
finetune.run_finetuning()

# 시각화용 모델 로드
ModelClass = {
    "GIN": "models.gin.GINNet",
    "GCN": "models.gcn.GCNNet",
    "GAT": "models.gat.GATNet"
}[cfg.MODEL_TYPE]
module, cls = ModelClass.rsplit(".", 1)
Model = getattr(__import__(module, fromlist=[cls]), cls)

_, _, tmp = load_data(
    cfg.TARGET_DATA_PATH,
    train_ratio=0, val_ratio=0, test_ratio=1,
    random_seed=cfg.RANDOM_SEED
)
input_dim = tmp[0].x.shape[1]
model = Model(
    input_dim=input_dim,
    hidden_dim=cfg.HIDDEN_DIM,
    output_dim=cfg.NUM_CLASSES,
    num_layers=cfg.NUM_LAYERS,
    dropout=cfg.DROPOUT,
    **({"heads": cfg.NUM_HEADS} if cfg.MODEL_TYPE.upper() == "GAT" else {})
)
load_model(model, cfg.FINETUNED_MODEL_PATH, device='cpu')

# 테스트 데이터 준비 및 시각화
_, _, test_data = load_data(
    cfg.TARGET_DATA_PATH,
    train_ratio=cfg.TRAIN_RATIO,
    val_ratio=cfg.VAL_RATIO,
    test_ratio=cfg.TEST_RATIO,
    random_seed=cfg.RANDOM_SEED
)
visualize_embeddings(model, test_data, save_path="embeddings.png")
print("✅ Saved embeddings.png")
PYCODE

echo "✅ All steps in unittest completed!"
