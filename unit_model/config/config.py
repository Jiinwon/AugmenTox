import os

## Data paths
SOURCE_DATA_PATH = os.path.join("data", "er_alpha.csv")   # Source dataset (ER alpha)
TARGET_DATA_PATH = os.path.join("data", "er_beta.csv")    # Target dataset (ER beta)

## Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

## Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS_PRETRAIN = 50
NUM_EPOCHS_FINETUNE = 30
LEARNING_RATE = 0.001
LR_STEP_SIZE = 10            # StepLR scheduler step
LR_GAMMA = 0.5               # Learning rate decay factor

## Model hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.5
MODEL_TYPE = "GIN"           # default model type ("GIN", "GCN", or "GAT")
NUM_CLASSES = 1              # output classes (1 for binary classification)
NUM_HEADS = 4               # number of heads for GAT (if applicable)

# File paths for saving models
PRETRAINED_MODEL_PATH = "pretrained_model.pth"
FINETUNED_MODEL_PATH = "finetuned_model.pth"
