import pandas as pd

# 원본 엑셀 불러오기
df = pd.read_excel("data/sample/ToxCast_v.4.2_mc_hitc_ER_sample.xlsx")

# 1) Pre-training용 CSV 생성
df_pre = df[["SMILES", "TOX21_ERa_BLA_Antagonist_ratio"]].dropna()
df_pre.columns = ["smiles", "label"]
df_pre.to_csv("data/sample/era_pretraining.csv", index=False)

# 2) Fine-tuning용 CSV 생성
df_fine = df[["SMILES", "TOX21_ERa_LUC_VM7_Antagonist_0.1nM_E2"]].dropna()
df_fine.columns = ["smiles", "label"]
df_fine.to_csv("data/sample/era_finetuning.csv", index=False)