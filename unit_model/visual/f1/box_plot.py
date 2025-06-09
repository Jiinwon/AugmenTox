import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 엑셀 파일 경로
excel_path = "./visual/performance_summary.xlsx"
df = pd.read_excel(excel_path)

# 유효한 diff 값만 필터링
df_filtered = df[df["diff"].notna() & df["target"].notna() & (df["target"] != "")]

# 모델 타입 정렬 (원하는 순서로 바꿔도 됨)
model_order = ["GIN", "GCN", "GAT", "GIN_GCN", "GIN_GAT", "GCN_GAT"]
df_filtered = df_filtered[df_filtered["model_type"].isin(model_order)]

# 시각화 스타일 설정
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered, x="model_type", y="diff", order=model_order, palette="Set2")

# 시각적 디테일
plt.title("F1(finetune) - F1(target-only) by Model Type")
plt.xlabel("Model Type")
plt.ylabel("F1 Difference")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()

# 저장
output_path = "./visual/f1_diff_boxplot_by_model.png"
plt.savefig(output_path, dpi=300)
plt.close()
print(f"✅ Box plot 저장 완료: {output_path}")