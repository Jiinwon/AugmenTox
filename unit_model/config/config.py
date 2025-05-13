import os
import torch

# Assay names
SOURCE_NAME = "TOX21_ERa_BLA_Antagonist_ratio"
TARGET_NAME = "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2"

## Data paths
DATA_BASE_PATH = os.path.join("data", "sample")
ENTIRE_DATA_PATH = os.path.join(DATA_BASE_PATH "ToxCast_v.4.2_mc_hitc_ER_sample.xlsx")
SOURCE_DATA_PATH = os.path.join(DATA_BASE_PATH, "pretraining.csv")    
TARGET_DATA_PATH = os.path.join(DATA_BASE_PATH, "finetuning.csv")    


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
NUM_CLASSES = 1              # output classes (1 for binary classification)
NUM_HEADS = 4               # number of heads for GAT (if applicable)

# File paths for saving models
PRETRAINED_MODEL_PATH = "pretrained_model.pth"
FINETUNED_MODEL_PATH = "finetuned_model.pth"

# 모델 체크포인트를 저장할 디렉토리
CHECKPOINT_DIR = "checkpoints"

# 체크포인트 디렉토리가 없으면 자동 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 사전학습된 모델 가중치
PRETRAINED_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pretrained_{}.pth".format(MODEL_TYPE.lower()))

# 파인튜닝된 모델 가중치
FINETUNED_MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "finetuned_{}.pth".format(MODEL_TYPE.lower()))
