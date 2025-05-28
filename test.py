import pandas as pd

# CSV 파일 경로
csv_path = "ToxCast_v.4.2_mc_hitc_ER.csv"

# 데이터 로딩
df = pd.read_csv(csv_path)

# 결과 저장 리스트
results = []

# 모든 열을 대상으로 분석
target_cols = df.columns

# 각 열에 대해 0, 1, NaN 개수 계산
for col in target_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        zero_count = (df[col] == 0).sum()
        one_count = (df[col] == 1).sum()
        nonempty_count = zero_count + one_count
        nan_count = df[col].isna().sum()
        results.append((col, nonempty_count, zero_count, one_count, nan_count))

# 데이터프레임 생성 및 정렬
summary_df = pd.DataFrame(results, columns=["Column", "Nonempty", "Zero Count", "One Count", "NaN Count"])

# 이름 기준으로 우선 정렬 (비슷한 이름끼리), 그 다음 NaN Count 기준 정렬
summary_df_sorted = summary_df.sort_values(by=["NaN Count", "Column"], ascending=[True, True])

# 엑셀 파일로 저장
summary_df_sorted.to_excel("toxcast_ER_summary_sorted.xlsx", index=False)

print("✅ 정렬된 요약 엑셀 파일이 생성되었습니다: toxcast_ER_summary_sorted.xlsx")