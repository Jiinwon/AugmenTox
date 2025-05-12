#!/bin/bash
#SBATCH --job-name=GNN_pipeline
#SBATCH --output=./log/slurm_%x_%j_%Y%m%d_%H%M.out
#SBATCH --error=./log/slurm_%x_%j_%Y%m%d_%H%M.err
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

# 로그 디렉토리 생성
LOG_DIR="./log"
mkdir -p "$LOG_DIR"
NOW=$(date +"%Y%m%d_%H%M")

# self-nohup 실행 (최초 실행 시에만)
if [ -z "$NOHUP_MODE" ]; then
    echo "[*] Launching via nohup..."
    NOHUP_MODE=1 nohup sbatch "$0" > "$LOG_DIR/unittest_${NOW}.log" 2>&1 &
    exit 0
fi

# 0. PYTHONPATH 설정
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# (필요시) 가상환경 활성화
# source /home1/won0316/anaconda3/envs/toxcast_env/bin/activate

# 1. Excel to CSV
echo "[1/4] Converting Excel to CSV..."
python -m scripts.convert_excel_to_csv

# 2~4. Pretraining + Finetuning + Embedding (각각 2에폭)
echo "[2/4] Pretraining on ERα..."
echo "[3/4] Finetuning on ERβ..."
echo "[4/4] Visualizing embeddings..."

python -u - << 'PYCODE'
import config.config as cfg
from data.load_data import load_data
from train import pretrain, finetune
from train.utils import load_model
from eval.visualize import visualize_embeddings

#cfg.NUM_EPOCHS_PRETRAIN = 2
#cfg.NUM_EPOCHS_FINETUNE = 2

pretrain.run_pretraining()
finetune.run_finetuning()

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

echo "✅ All steps in SBATCH pipeline completed!"
