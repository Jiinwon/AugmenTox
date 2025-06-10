#!/usr/bin/env python3
"""
visualize_results.py

집계된 CV 결과 JSON 파일을 읽어 F1 score 분포를 히스토그램과 박스플롯으로 비교하고
이미지 파일로 저장하는 스크립트

Usage:
    python visualize_results.py \
        --json1 results/modelA_agg.json \
        --json2 results/modelB_agg.json \
        --labels ModelA ModelB \
        --out-dir figures

출력:
    figures/f1_distribution_comparison.png
"""
import sys, os
# 상위 디렉토리 경로 추가 (config나 기타 모듈 필요시)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 수정필요

import argparse
import json
import matplotlib.pyplot as plt  # matplotlib 설치 필요


def main():
    parser = argparse.ArgumentParser(description='F1 분포 시각화')
    parser.add_argument('--json1', required=True, help='모델 A 집계 결과 JSON')  # 수정필요: 파일명 확인
    parser.add_argument('--json2', required=True, help='모델 B 집계 결과 JSON')  # 수정필요
    parser.add_argument('--labels', nargs=2, default=['ModelA', 'ModelB'], help='플롯 레이블')
    parser.add_argument('--out-dir', required=True, help='이미지 저장 디렉토리')
    args = parser.parse_args()

    # JSON 로드
    with open(args.json1) as f1:
        dist1 = json.load(f1).get('dist', [])
    with open(args.json2) as f2:
        dist2 = json.load(f2).get('dist', [])

    # 디렉토리 생성
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'f1_distribution_comparison.png')

    # 플롯 생성
    plt.figure(figsize=(10, 5))

    # 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(dist1, bins=20, alpha=0.6, label=args.labels[0])
    plt.hist(dist2, bins=20, alpha=0.6, label=args.labels[1])
    plt.title('F1 Score Distribution')
    plt.xlabel('F1 Score')
    plt.ylabel('Frequency')
    plt.legend()

    # 박스플롯
    plt.subplot(1, 2, 2)
    plt.boxplot([dist1, dist2], labels=args.labels)
    plt.title('F1 Score Boxplot')
    plt.ylabel('F1 Score')

    # 저장
    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Visualization saved to {out_path}')

if __name__ == '__main__':
    main()
