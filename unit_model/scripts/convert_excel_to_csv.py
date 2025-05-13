import pandas as pd
import config.config as cfg
import os

# 환경변수로부터 이름 읽기 (없으면 config 기본값 사용)
pretrain_name = os.getenv("SOURCE_NAME", cfg.SOURCE_NAME)
finetune_name = os.getenv("TARGET_NAME", cfg.TARGET_NAME)

# 원본 데이터 경로
data_path = cfg.ENTIRE_DATA_PATH

# 동적 디렉토리 생성: data/sample/SOURCE&&TARGET/
combo_dir = os.path.join("data", "sample", f"{pretrain_name}&&{finetune_name}")
os.makedirs(combo_dir, exist_ok=True)

# pretrain/finetune 경로를 해당 디렉토리 아래로 변경
pretrain_path = os.path.join(combo_dir, "pretraining.csv")
finetune_path = os.path.join(combo_dir, "finetuning.csv")

# 원본 엑셀 불러오기
df = pd.read_excel(data_path)

# 1) Pre-training용 CSV 생성
df_pre = df[["SMILES", pretrain_name]].dropna()
df_pre.columns = ["smiles", "label"]
df_pre.to_csv(pretrain_path, index=False)

# 2) Fine-tuning용 CSV 생성
df_fine = df[["SMILES", finetune_name]].dropna()
df_fine.columns = ["smiles", "label"]
df_fine.to_csv(finetune_path, index=False)