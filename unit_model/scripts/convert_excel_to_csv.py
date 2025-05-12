import pandas as pd
import config.config as cfg

data_path = cfg.ENTIRE_DATA_PATH
pretrain_path = cfg.SOURCE_DATA_PATH
pretrain_name = cfg.SOURCE_NAME
finetune_path = cfg.TARGET_DATA_PATH
finetune_name = cfg.TARGET_NAME

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