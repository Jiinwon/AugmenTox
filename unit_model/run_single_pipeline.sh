#!/bin/bash

set -e

# 1. 로그 디렉토리 설정
mkdir -p "$LOG_SUBDIR"
LOG_OUT="$LOG_SUBDIR/${SOURCE_NAME}&&${TARGET_NAME}_${MODEL_TYPE}.out"
LOG_ERR="$LOG_SUBDIR/${SOURCE_NAME}&&${TARGET_NAME}_${MODEL_TYPE}.err"
exec > "$LOG_OUT" 2> "$LOG_ERR"

# 2. PYTHONPATH 설정
DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$DIR"

# 3. 가상환경 활성화 (필요시)
# source /home1/USER/anaconda3/envs/toxcast_env/bin/activate

# 4. convert_excel_to_csv: 변환 수행
# if [ "$OPERA" = "False" ]; then
#     echo "[1/4] Excel → CSV 변환 중..."
#     python3 -u -m scripts.convert_excel_to_csv
# fi

# 5. pretrain
echo "[2/4] Pretraining on $SOURCE_NAME using $MODEL_TYPE..."
python3 -u - <<PYCODE
import os
import config.config as cfg
cfg.SOURCE_NAME = os.environ["SOURCE_NAME"]
cfg.MODEL_TYPE = os.environ["MODEL_TYPE"]
# OPERA 모드 시 SDF 사용
if cfg.OPERA:
    cfg.SOURCE_DATA_PATH = cfg.SOURCE_SDF_PATH
from train.pretrain import run_pretraining
run_pretraining()
PYCODE

# 6. finetune
echo "[3/4] Finetuning on $TARGET_NAME using $MODEL_TYPE..."
python3 -u - <<PYCODE
import os
import config.config as cfg
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE = os.environ["MODEL_TYPE"]
from train.finetune import run_finetuning
run_finetuning()
PYCODE

# 7. target-only
echo "[3.1/4] Target-only training on $TARGET_NAME using $MODEL_TYPE..."
python3 -u - <<PYCODE
import os
import config.config as cfg
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE = os.environ["MODEL_TYPE"]
from train.target_only import run_target_only
run_target_only()
PYCODE

# 8. visualize
echo "[4/4] Embedding 시각화 중..."
python3 -u - <<PYCODE
import os, torch
import config.config as cfg
from models import gin, gcn, gat
from models import gin_gcn, gin_gat, gcn_gat
from data.load_data import load_data
from train.utils import load_model
from eval.visualize import visualize_embeddings

cfg.SOURCE_NAME = os.environ["SOURCE_NAME"]
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]

ModelClass = {
    "GIN": gin.GINNet,
    "GCN": gcn.GCNNet,
    "GAT": gat.GATNet,
    "GIN_GCN": gin_gcn.GIN_GCN_Hybrid,
    "GIN_GAT": gin_gat.GIN_GAT_Hybrid,
    "GCN_GAT": gcn_gat.GCN_GAT_Hybrid,
}[cfg.MODEL_TYPE]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 입력 차원 확보용 데이터
_, _, all_test = load_data(cfg.TARGET_DATA_PATH, train_ratio=0, val_ratio=0, test_ratio=1, random_seed=cfg.RANDOM_SEED)
input_dim = all_test[0].x.shape[1]

# 모델 생성 (GAT 포함 여부에 따라 heads 인자 유무 조정)
kwargs = {}
if "GAT" in cfg.MODEL_TYPE:
    kwargs["heads"] = cfg.NUM_HEADS

model = ModelClass(
    input_dim=input_dim,
    hidden_dim=cfg.HIDDEN_DIM,
    output_dim=cfg.NUM_CLASSES,
    num_layers=cfg.NUM_LAYERS,
    dropout=cfg.DROPOUT,
    **kwargs
)
load_model(model, cfg.FINETUNED_MODEL_PATH, device)


# 시각화 수행
_, _, test_data = load_data(cfg.TARGET_DATA_PATH, train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO, random_seed=cfg.RANDOM_SEED)

# 저장 경로: unit_model/figure/MODEL_TYPE/파일명.png
save_dir = os.path.join("figure", cfg.MODEL_TYPE)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir,
    f"{embeddings_cfg.SOURCE_NAME.replace('/', '_')}_{cfg.TARGET_NAME.replace('/', '_')}_{cfg.MODEL_TYPE}.png"
)

visualize_embeddings(model, test_data, device, save_path=save_path)
print(f"✅ Embeddings 저장 완료: {save_path}")
PYCODE

echo "✅ 실험 완료: ${SOURCE_NAME} -> ${TARGET_NAME} [${MODEL_TYPE}]"