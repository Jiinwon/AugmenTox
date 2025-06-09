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
: <<'COMMENT'
# 4. convert_excel_to_csv: 변환 수행
if [ "$OPERA" = "False" ]; then
    echo "[1/4] Excel → CSV 변환 중..."
    python3 -u -m scripts.convert_excel_to_csv
fi
COMMENT

# 5. pretrain
echo "[2/4] Pretraining on $SOURCE_NAME using $MODEL_TYPE..."
# config.PRETRAINED_MODEL_PATH already includes BASE_SAVE_DIR/pretrained/{MODEL_NAME}/Pretrained_{SOURCE_NAME}_{MODEL_NAME}.pth
PRETRAIN_FILE=$(python3 - <<PYCODE
import os, config.config as cfg
print(cfg.PRETRAINED_MODEL_PATH)
PYCODE
)
if [ -f "$PRETRAIN_FILE" ]; then
    echo "  * Pretrained model already exists at $PRETRAIN_FILE, loading..."
else
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
fi
#: <<'COMMENT'
# 6. finetune
echo "[3/4] Finetuning on $TARGET_NAME using $MODEL_TYPE..."
# finetuned 모델 경로 설정
FINETUNE_PATH=$(python3 - <<PYCODE
import os, config.config as cfg
cfg.SOURCE_NAME = os.environ["SOURCE_NAME"]
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]
print(cfg.FINETUNED_MODEL_PATH)
PYCODE
)
if [ -f "$FINETUNE_PATH" ]; then
    echo "  * 파인튜닝된 모델 이미 존재: $FINETUNE_PATH"
    echo "  * Finetune 생략, 저장된 모델로 평가만 진행"
    python3 -u - <<EVAL_FT
import os, torch
import config.config as cfg
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from train.utils import load_model, evaluate

# 환경 변수
cfg.SOURCE_NAME = os.environ["SOURCE_NAME"]
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]

# 테스트 데이터 로드
_, _, test_data = load_data(
    cfg.TARGET_DATA_PATH,
    train_ratio=cfg.TRAIN_RATIO,
    val_ratio=cfg.VAL_RATIO,
    test_ratio=cfg.TEST_RATIO,
    random_seed=cfg.RANDOM_SEED
)
input_dim = test_data[0].x.shape[1]
output_dim = cfg.NUM_CLASSES

# 모델 생성
mt = cfg.MODEL_TYPE.upper()
if mt == "GIN":
    from models.gin import GINNet
    model = GINNet(input_dim, cfg.HIDDEN_DIM, output_dim,
                   num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
elif mt == "GCN":
    from models.gcn import GCNNet
    model = GCNNet(input_dim, cfg.HIDDEN_DIM, output_dim,
                   num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
elif mt == "GAT":
    from models.gat import GATNet
    model = GATNet(input_dim, cfg.HIDDEN_DIM, output_dim,
                   num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
elif mt == "GIN_GCN":
    from models.gin_gcn import GIN_GCN_Hybrid as M
    model = M(input_dim, cfg.HIDDEN_DIM, output_dim,
              num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
elif mt == "GIN_GAT":
    from models.gin_gat import GIN_GAT_Hybrid as M
    model = M(input_dim, cfg.HIDDEN_DIM, output_dim,
              num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
elif mt == "GCN_GAT":
    from models.gcn_gat import GCN_GAT_Hybrid as M
    model = M(input_dim, cfg.HIDDEN_DIM, output_dim,
              num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
else:
    raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 저장된 파인튜닝 모델 불러오기
load_model(model, cfg.FINETUNED_MODEL_PATH, device)

# 테스트 데이터로 최종 성능 평가
test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
test_metrics = evaluate(model, test_loader, device)
print("=== Fine-tuned TEST performance ===")
print(f"Test F1: {test_metrics['f1']:.4f}, "
      f"ROC-AUC: {test_metrics['roc_auc']:.4f}, "
      f"PR-AUC: {test_metrics['pr_auc']:.4f}")
EVAL_FT
else
    python3 -u - <<PYCODE
import os
import config.config as cfg
cfg.SOURCE_NAME = os.environ["SOURCE_NAME"]
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]
from train.finetune import run_finetuning
run_finetuning()
PYCODE
fi
 

# 7. target-only
echo "[3.1/4] Target-only training on $TARGET_NAME using $MODEL_TYPE..."
# 타겟-온리 모델 경로 확인
TARGETONLY_PATH=$(python3 - <<PYCODE
import os, config.config as cfg
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]
print(cfg.TARGET_ONLY_MODEL_PATH)
PYCODE
)
if [ -f "$TARGETONLY_PATH" ]; then
    echo "  * Target-only 모델 이미 존재: $TARGETONLY_PATH"
    echo "  * Target-only 생략, 저장된 모델로 평가만 진행"
    python3 -u - <<EVAL_TO
import os, torch
import config.config as cfg
from torch_geometric.loader import DataLoader
from data.load_data import load_data
from train.utils import load_model, evaluate

# 환경 변수
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]

# 테스트 데이터 로드
_, _, test_data = load_data(
    cfg.TARGET_DATA_PATH,
    train_ratio=cfg.TRAIN_RATIO,
    val_ratio=cfg.VAL_RATIO,
    test_ratio=cfg.TEST_RATIO,
    random_seed=cfg.RANDOM_SEED
)
input_dim = test_data[0].x.shape[1]
output_dim = cfg.NUM_CLASSES

# 모델 생성 (Target-only 구조가 Fine-tune과 동일 가정)
mt = cfg.MODEL_TYPE.upper()
if mt == "GIN":
    from models.gin import GINNet
    model = GINNet(input_dim, cfg.HIDDEN_DIM, output_dim,
                   num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
elif mt == "GCN":
    from models.gcn import GCNNet
    model = GCNNet(input_dim, cfg.HIDDEN_DIM, output_dim,
                   num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
elif mt == "GAT":
    from models.gat import GATNet
    model = GATNet(input_dim, cfg.HIDDEN_DIM, output_dim,
                   num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
elif mt == "GIN_GCN":
    from models.gin_gcn import GIN_GCN_Hybrid as M
    model = M(input_dim, cfg.HIDDEN_DIM, output_dim,
              num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
elif mt == "GIN_GAT":
    from models.gin_gat import GIN_GAT_Hybrid as M
    model = M(input_dim, cfg.HIDDEN_DIM, output_dim,
              num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
elif mt == "GCN_GAT":
    from models.gcn_gat import GCN_GAT_Hybrid as M
    model = M(input_dim, cfg.HIDDEN_DIM, output_dim,
              num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
else:
    raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 저장된 Target-only 모델 불러오기
load_model(model, cfg.TARGET_ONLY_MODEL_PATH, device)

# 테스트 데이터로 최종 성능 평가
test_loader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
test_metrics = evaluate(model, test_loader, device)
print("=== Target-only TEST performance ===")
print(f"Test F1: {test_metrics['f1']:.4f}, "
      f"ROC-AUC: {test_metrics['roc_auc']:.4f}, "
      f"PR-AUC: {test_metrics['pr_auc']:.4f}")
EVAL_TO
else
    python3 -u - <<PYCODE
import os
import config.config as cfg
cfg.TARGET_NAME = os.environ["TARGET_NAME"]
cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]
from train.target_only import run_target_only
run_target_only()
PYCODE
fi
#COMMENT



# 8. visualize
# echo "[4/4] Embedding 시각화 중..."
# python3 -u - <<PYCODE
# import os, torch
# import config.config as cfg
# from models import gin, gcn, gat
# from models import gin_gcn, gin_gat, gcn_gat
# from data.load_data import load_data
# from train.utils import load_model
# from eval.visualize import visualize_embeddings

# cfg.SOURCE_NAME = os.environ["SOURCE_NAME"]
# cfg.TARGET_NAME = os.environ["TARGET_NAME"]
# cfg.MODEL_TYPE  = os.environ["MODEL_TYPE"]

# ModelClass = {
#     "GIN": gin.GINNet,
#     "GCN": gcn.GCNNet,
#     "GAT": gat.GATNet,
#     "GIN_GCN": gin_gcn.GIN_GCN_Hybrid,
#     "GIN_GAT": gin_gat.GIN_GAT_Hybrid,
#     "GCN_GAT": gcn_gat.GCN_GAT_Hybrid,
# }[cfg.MODEL_TYPE]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # 입력 차원 확보용 데이터
# _, _, all_test = load_data(cfg.TARGET_DATA_PATH, train_ratio=0, val_ratio=0, test_ratio=1, random_seed=cfg.RANDOM_SEED)
# input_dim = all_test[0].x.shape[1]

# # 모델 생성 (GAT 포함 여부에 따라 heads 인자 유무 조정)
# kwargs = {}
# if "GAT" in cfg.MODEL_TYPE:
#     kwargs["heads"] = cfg.NUM_HEADS

# model = ModelClass(
#     input_dim=input_dim,
#     hidden_dim=cfg.HIDDEN_DIM,
#     output_dim=cfg.NUM_CLASSES,
#     num_layers=cfg.NUM_LAYERS,
#     dropout=cfg.DROPOUT,
#     **kwargs
# )
# load_model(model, cfg.FINETUNED_MODEL_PATH, device)


# # 시각화 수행
# _, _, test_data = load_data(cfg.TARGET_DATA_PATH, train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO, random_seed=cfg.RANDOM_SEED)

# # 저장 경로: unit_model/figure/MODEL_TYPE/파일명.png
# save_dir = os.path.join("figure", cfg.MODEL_TYPE)
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(
#     save_dir,
#     f"{cfg.SOURCE_NAME.replace('/', '_')}_{cfg.TARGET_NAME.replace('/', '_')}_{cfg.MODEL_TYPE}.png"
# )

# visualize_embeddings(model, test_data, device, save_path=save_path)
# print(f"✅ Embeddings 저장 완료: {save_path}")
# PYCODE

echo "✅ 실험 완료: ${SOURCE_NAME} -> ${TARGET_NAME} [${MODEL_TYPE}]"