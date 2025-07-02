import os
import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 파일 경로
excel_path = "./visual/performance_summary.xlsx"
df = pd.read_excel(excel_path)

# 출력 디렉토리 설정
output_dir = "./visual/heatmap_by_model"
os.makedirs(output_dir, exist_ok=True)

# 모델 리스트
model_types = df["model_type"].dropna().unique()

for model in model_types:
    # 해당 모델에 해당하는 행 필터링
    df_model = df[(df["model_type"] == model) & df["target"].notna() & (df["target"] != "")]

    if df_model.empty:
        print(f"⚠️ 데이터 없음: {model}")
        continue

    # 중복 제거 (같은 source-target 쌍이 여럿 있는 경우 평균)
    df_grouped = df_model.groupby(["source", "target"], as_index=False)["diff"].mean()

    # 피벗 테이블 생성
    heatmap_data = df_grouped.pivot(index="source", columns="target", values="diff")

    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap_data, aspect="auto", cmap="viridis")
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=90)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.title(f"F1(finetune) - F1(target-only)\nModel: {model}")
    plt.colorbar(im, label="F1_diff")
    plt.tight_layout()

    # 저장
    fname = f"heatmap_diff_{model}.png"
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 저장 완료: {fpath}")