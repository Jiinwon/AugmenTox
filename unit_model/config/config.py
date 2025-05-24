import os
import torch

# Assay names
SOURCE_NAMES = [
    "TOX21_ERb_BLA_Antagonist_ch1",
    "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2_viability",
    "TOX21_ERa_LUC_VM7_Agonist",
    "TOX21_ERa_BLA_Antagonist_ch1",
]
SOURCE_NAME = os.getenv("SOURCE_NAME", "TOX21_ERa_BLA_Agonist_ch1")

TARGET_NAMES = [
    "TOX21_ERb_BLA_Agonist_ch2",
    "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780",
    "TOX21_ERa_BLA_Agonist_ch2",
    "TOX21_ERb_BLA_Antagonist_viability",
]
TARGET_NAME = os.getenv("TARGET_NAME", "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2")



## Data paths
DATA_BASE_PATH = os.path.join("data", "sample")
ENTIRE_DATA_PATH = os.path.join(DATA_BASE_PATH, "ToxCast_v.4.2_mc_hitc_ER.xlsx")

# 조합 디렉토리: data/sample/{SOURCE_NAME}&&{TARGET_NAME}/
COMBO_DIR = os.path.join(DATA_BASE_PATH, f"{SOURCE_NAME}&&{TARGET_NAME}")
# CSV 경로는 조합 디렉토리 내로 설정
SOURCE_DATA_PATH = os.path.join(COMBO_DIR, "pretraining.csv")
TARGET_DATA_PATH = os.path.join(COMBO_DIR, "finetuning.csv")  


## Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

## Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS_PRETRAIN = 50
NUM_EPOCHS_FINETUNE = 50  # 원래는 30
LEARNING_RATE = 0.001
LR_STEP_SIZE = 10            # StepLR scheduler step
LR_GAMMA = 0.5               # Learning rate decay factor

## Model hyperparameters
HIDDEN_DIM = 1024  # 원래는 64
NUM_LAYERS = 2  # 원래는 3 
DROPOUT = 0.5
MODEL_TYPE = "GIN"           # default model type ("GIN", "GCN", or "GAT")
MODEL_NAME = MODEL_TYPE.lower()
NUM_CLASSES = 1              # output classes (1 for binary classification)
NUM_HEADS = 4               # number of heads for GAT (if applicable)

# 저장 베이스 디렉토리
BASE_SAVE_DIR = "model_save"

# 모델 저장 디렉토리 경로: model_save/gin/
MODEL_DIR = os.path.join(BASE_SAVE_DIR, MODEL_NAME)

# 디렉토리 생성
os.makedirs(MODEL_DIR, exist_ok=True)

# 저장 경로 구성
PRETRAINED_MODEL_PATH   = os.path.join(MODEL_DIR, f"{SOURCE_NAME}&&{SOURCE_NAME}_{MODEL_NAME}.pth")
FINETUNED_MODEL_PATH    = os.path.join(MODEL_DIR, f"{SOURCE_NAME}&&{TARGET_NAME}_{MODEL_NAME}.pth")
TARGET_ONLY_MODEL_PATH  = os.path.join(MODEL_DIR, f"{TARGET_NAME}&&{TARGET_NAME}_{MODEL_NAME}.pth")