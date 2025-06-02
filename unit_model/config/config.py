import os
import torch

# Assay names
# SOURCE_NAMES = [
#     "TOX21_ERa_LUC_VM7_Agonist",                            # nonempty : 8305, zero : 7243, one : 1062
#     # "TOX21_ERa_BLA_Antagonist_viability",                   # nonempty : 8305, zero : 7990, one : 315
# ]
# SOURCE_NAME = os.getenv("SOURCE_NAME", "TOX21_ERa_LUC_VM7_Agonist")


# TARGET_NAMES = [
#     "TOX21_ERb_BLA_Antagonist_ratio",                       # nonempty : 7871, zero : 6406, one : 1465
#     # "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780",             # nonempty : 7871, zero : 7712, one : 159
# ]
# TARGET_NAME = os.getenv("TARGET_NAME", "TOX21_ERb_BLA_Antagonist_ratio")

SOURCE_NAMES = [
     "TOX21_ERa_BLA_Antagonist_ratio",                       # nonempty : 8305, zero : 7244, one : 1061
     "TOX21_ERa_LUC_VM7_Agonist",                            # nonempty : 8305, zero : 7243, one : 1062
     "TOX21_ERa_BLA_Antagonist_viability",                   # nonempty : 8305, zero : 7990, one : 315
     "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2",                # nonempty : 8305, zero : 7468, one : 837
     "TOX21_ERa_BLA_Antagonist_ch2",                         # nonempty : 8305, zero : 7830, one : 475
     "TOX21_ERa_BLA_Antagonist_ch1",                         # nonempty : 8305, zero : 7774, one : 531
     "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2_viability",      # nonempty : 8305, zero : 7707, one : 598
     "TOX21_ERa_BLA_Agonist_ch2",                            # nonempty : 8305, zero : 7885, one : 420
     "TOX21_ERa_BLA_Agonist_ch1",                            # nonempty : 8305, zero : 7912, one : 393
     "TOX21_ERa_BLA_Agonist_ratio",                          # nonempty : 8305, zero : 7849, one : 456
     "TOX21_ERb_BLA_Antagonist_ch1",                         # nonempty : 7871, zero : 7627, one : 244
     "TOX21_ERb_BLA_Antagonist_ch2",                         # nonempty : 7871, zero : 7058, one : 813
     "TOX21_ERb_BLA_Agonist_viability",                      # nonempty : 7871, zero : 6782, one : 1089
     "TOX21_ERb_BLA_Agonist_ratio",                          # nonempty : 7871, zero : 7224, one : 147
     "TOX21_ERb_BLA_Agonist_ch2",                            # nonempty : 7871, zero : 7692, one : 179
     "TOX21_ERb_BLA_Agonist_ch1",                            # nonempty : 7871, zero : 7735, one : 136
     "TOX21_ERR_LUC_Agonist",                                # nonempty : 7871, zero : 7622, one : 249
     "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780_viability",   # nonempty : 7871, zero : 7164, one : 707
     "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2",                # nonempty : 7871, zero : 6889, one : 982
     "TOX21_ERb_BLA_Antagonist_ratio",                       # nonempty : 7871, zero : 6406, one : 1465
     "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780",             # nonempty : 7871, zero : 7712, one : 159
     "TOX21_ERR_LUC_viability",                              # nonempty : 7871, zero : 6819, one : 1052
     "TOX21_ERR_LUC_Antagonist",                             # nonempty : 7871, zero : 6245, one : 1626
     "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2_viability",      # nonempty : 7871, zero : 7306, one : 565
     "TOX21_ERb_BLA_Antagonist_viability",                   # nonempty : 7871, zero : 6811, one : 1060
 ]
SOURCE_NAME = os.getenv("SOURCE_NAME", "TOX21_ERa_BLA_Antagonist_ratio")

TARGET_NAMES = [
     "TOX21_ERa_BLA_Antagonist_ratio",                       # nonempty : 8305, zero : 7244, one : 1061
     "TOX21_ERa_LUC_VM7_Agonist",                            # nonempty : 8305, zero : 7243, one : 1062
     "TOX21_ERa_BLA_Antagonist_viability",                   # nonempty : 8305, zero : 7990, one : 315
     "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2",                # nonempty : 8305, zero : 7468, one : 837
     "TOX21_ERa_BLA_Antagonist_ch2",                         # nonempty : 8305, zero : 7830, one : 475
     "TOX21_ERa_BLA_Antagonist_ch1",                         # nonempty : 8305, zero : 7774, one : 531
     "TOX21_ERa_LUC_VM7_Antagonist_0.5nM_E2_viability",      # nonempty : 8305, zero : 7707, one : 598
     "TOX21_ERa_BLA_Agonist_ch2",                            # nonempty : 8305, zero : 7885, one : 420
     "TOX21_ERa_BLA_Agonist_ch1",                            # nonempty : 8305, zero : 7912, one : 393
     "TOX21_ERa_BLA_Agonist_ratio",                          # nonempty : 8305, zero : 7849, one : 456
     "TOX21_ERb_BLA_Antagonist_ch1",                         # nonempty : 7871, zero : 7627, one : 244
     "TOX21_ERb_BLA_Antagonist_ch2",                         # nonempty : 7871, zero : 7058, one : 813
     "TOX21_ERb_BLA_Agonist_viability",                      # nonempty : 7871, zero : 6782, one : 1089
     "TOX21_ERb_BLA_Agonist_ratio",                          # nonempty : 7871, zero : 7224, one : 147
     "TOX21_ERb_BLA_Agonist_ch2",                            # nonempty : 7871, zero : 7692, one : 179
     "TOX21_ERb_BLA_Agonist_ch1",                            # nonempty : 7871, zero : 7735, one : 136
     "TOX21_ERR_LUC_Agonist",                                # nonempty : 7871, zero : 7622, one : 249
     "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780_viability",   # nonempty : 7871, zero : 7164, one : 707
     "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2",                # nonempty : 7871, zero : 6889, one : 982
     "TOX21_ERb_BLA_Antagonist_ratio",                       # nonempty : 7871, zero : 6406, one : 1465
     "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780",             # nonempty : 7871, zero : 7712, one : 159
     "TOX21_ERR_LUC_viability",                              # nonempty : 7871, zero : 6819, one : 1052
     "TOX21_ERR_LUC_Antagonist",                             # nonempty : 7871, zero : 6245, one : 1626
     "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2_viability",      # nonempty : 7871, zero : 7306, one : 565
     "TOX21_ERb_BLA_Antagonist_viability",                   # nonempty : 7871, zero : 6811, one : 1060
 ]
TARGET_NAME = os.getenv("TARGET_NAME", "TOX21_ERa_BLA_Antagonist_ratio")


OPERA = False
# SDF에 들어 있는 모든 클래스 필드 순서
SDF_LABEL_FIELDS = ["Agonist_Class", "Antagonist_Class", "Binding_Class"]
ENDPOINTS = ["Binding_Class"]  # 원하는 대로 토글


## Data paths
DATA_BASE_PATH = os.path.join("data", "sample")
ENTIRE_DATA_PATH = os.path.join(DATA_BASE_PATH, "ToxCast_v.4.2_mc_hitc_ER.xlsx")

# 조합 디렉토리: data/sample/{SOURCE_NAME}&&{TARGET_NAME}/
#COMBO_DIR = os.path.join(DATA_BASE_PATH, f"{SOURCE_NAME}&&{TARGET_NAME}")
# CSV 경로는 조합 디렉토리 내로 설정
SOURCE_DATA_PATH = os.path.join(DATA_BASE_PATH, f"{SOURCE_NAME}.csv")
TARGET_DATA_PATH = os.path.join(DATA_BASE_PATH, f"{TARGET_NAME}.csv")  

# OPERA=True일 경우 사용할 SDF 파일 경로 지정 (예시는 기본값)
SOURCE_SDF_PATH = os.path.join(
    "..",
    "OPERA_data",
    "Supplemental_Material_2_TrainingSet.sdf"
)
if OPERA:
    # OPERA 모드면 SOURCE_DATA_PATH를 SDF 경로로 설정
    SOURCE_DATA_PATH = SOURCE_SDF_PATH


## Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

## Training hyperparameters
BATCH_SIZE = 32         # 32   
# NUM_EPOCHS_PRETRAIN = 50
NUM_EPOCHS_PRETRAIN = 100
NUM_EPOCHS_FINETUNE = 50
LEARNING_RATE = 0.001
LR_STEP_SIZE = 10            # StepLR scheduler step
LR_GAMMA = 0.5               # Learning rate decay factor

## Model hyperparameters
HIDDEN_DIM = 512  # 원래는 64
NUM_LAYERS = 2  # 원래는 3 
DROPOUT = 0.5  # 원래는 0.5


# 기존 모델 타입 + 하이브리드 타입 지원
MODEL_TYPE = os.getenv("MODEL_TYPE", "GIN_GCN")  # 가능 값: GIN, GCN, GAT, GIN_GCN, GIN_GAT, GCN_GAT
# 하이브리드 모델일 경우 파일명 표준화
MODEL_NAME = MODEL_TYPE.lower().replace("_", "_")

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
TARGET_ONLY_MODEL_PATH  = os.path.join(MODEL_DIR, f"TargetOnly_{TARGET_NAME}_{MODEL_NAME}.pth")