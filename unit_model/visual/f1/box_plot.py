import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 데이터 로드
df = pd.read_excel("./visual/performance_summary.xlsx")
df_filtered = df[df["diff"].notna()].copy()

# 저장 디렉토리
save_dir = "./visual/boxplot_horizontal_by_model"
os.makedirs(save_dir, exist_ok=True)

# 모델 리스트
model_types = df_filtered["model_type"].dropna().unique()

# seaborn 스타일
sns.set_style("whitegrid")

for model in model_types:
    sub_df = df_filtered[df_filtered["model_type"] == model].copy()

    # 'Supplemental' 처리
    sub_df["source"] = sub_df["source"].apply(
        lambda s: "OPERA" if "supplemental" in s.lower() else s
    )

    # source 개수에 따라 자동 높이 조정
    num_sources = sub_df["source"].nunique()
    fig_height = max(6, num_sources * 0.4)

    plt.figure(figsize=(10, fig_height))
    sns.boxplot(
        data=sub_df,
        y="source",   # assay name
        x="diff",     # F1 차이
        palette="Set1",
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="orange"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="black", markersize=4, linestyle='none', markeredgecolor="black")
    )
    
    plt.axvline(x=0.0, color="red", linestyle="--", linewidth=1.2)
    plt.title(f"F1 Score Gain by Assay (Model: {model})", fontsize=13)
    plt.xlabel("F1 Score Improvement (fine-tuned - target-only)", fontsize=11)
    plt.ylabel("Assay (Source Name)", fontsize=11)
    plt.tight_layout()

    # 저장
    save_path = os.path.join(save_dir, f"boxplot_horizontal_{model}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 저장 완료: {save_path}")