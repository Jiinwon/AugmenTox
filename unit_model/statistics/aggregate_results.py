#!/usr/bin/env python3
"""
aggregate_results.py

반복 K-Fold CV 결과 CSV를 읽어, F1 score의 평균, 표준편차, 분포를 계산하여 JSON 파일로 저장하는 스크립트

Usage:
    python aggregate_results.py \
        --input-csv results/gnn_scores.csv \
        --out-json results/agg_results.json
"""
import argparse
import pandas as pd  # 수정필요: pandas 설치 여부 확인
import json
import os

def main():
    parser = argparse.ArgumentParser(description='CV 결과 집계 스크립트')
    parser.add_argument('--input-csv', required=True, help='반복 CV 결과가 저장된 CSV 파일 경로')  # 수정필요: CSV 컬럼명 확인
    parser.add_argument('--out-json', required=True, help='출력 JSON 파일 경로')
    args = parser.parse_args()

    # CSV 읽기
    df = pd.read_csv(args.input_csv)
    if 'f1' not in df.columns:
        raise ValueError("입력 CSV에 'f1' 컬럼이 없습니다.")

    # 통계 계산
    f1_values = df['f1'].astype(float).tolist()
    mean_f1 = float(df['f1'].mean())
    std_f1 = float(df['f1'].std())

    # 결과 딕셔너리 구성
    results = {
        'mean': mean_f1,
        'std': std_f1,
        'dist': f1_values
    }

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # JSON 저장
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Aggregation results saved to {args.out_json}")

if __name__ == '__main__':
    main()
