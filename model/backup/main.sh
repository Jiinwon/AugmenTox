#!/usr/bin/env bash
set -e

# 0. 프로젝트 루트(이 스크립트가 있는 디렉토리)를 PYTHONPATH에 추가
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# (선택) 가상환경 활성화
# source /path/to/venv/bin/activate

# 1. Excel → CSV 변환
echo "[1/4] Converting Excel to CSV..."
python -m scripts.convert_excel_to_csv

# 2. Pre-training (ERα)
echo "[2/4] Pre-training on ERα..."
python -m train.pretrain

# 3. Fine-tuning (ERβ)
echo "[3/4] Fine-tuning on ERβ..."
python -m train.finetune

# 4. Embedding 시각화
echo "[4/4] Visualizing embeddings..."
python - << 'PYCODE'
from eval.visualize import visualize_embeddings
from train.utils import load_model
import config.config as cfg
from data.load_data import load_data

# 모델 로드
ModelClass = {"GIN": "models.gin.GINNet", "GCN": "models.gcn.GCNNet", "GAT": "models.gat.GATNet"}[cfg.MODEL_TYPE]
module, cls = ModelClass.rsplit(".", 1)
model = getattr(__import__(module, fromlist=[cls]), cls)(
    input_dim=load_data(cfg.TARGET_DATA_PATH, cfg.SMILES_COL, cfg.LABEL_COL_TARGET, 0,0,1, cfg.RANDOM_SEED)[0][0].x.shape[1],
    hidden_dim=cfg.HIDDEN_DIM,
    output_dim=cfg.NUM_CLASSES,
    num_layers=cfg.NUM_LAYERS,
    dropout=cfg.DROPOUT,
    heads=getattr(cfg, "NUM_HEADS", 1)
)
load_model(model, cfg.FINETUNED_MODEL_PATH, device='cpu')

# 테스트 데이터 준비
_, _, test_data = load_data(
    cfg.TARGET_DATA_PATH,
    cfg.SMILES_COL,
    cfg.LABEL_COL_TARGET,
    cfg.TRAIN_RATIO,
    cfg.VAL_RATIO,
    cfg.TEST_RATIO,
    cfg.RANDOM_SEED
)
visualize_embeddings(model, test_data, save_path="embeddings.png")
print("Saved embeddings.png")
PYCODE

echo "✅ All steps completed successfully!"