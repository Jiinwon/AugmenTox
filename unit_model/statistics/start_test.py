#!/usr/bin/env python3
"""
stat_test.py

두 모델의 CV 결과 JSON 파일을 읽어
paired t-test와 Wilcoxon signed-rank test를 수행하고
결과를 터미널 및 JSON으로 출력하는 스크립트

Usage:
    python stat_test.py \
        --json1 results/modelA_agg.json \
        --json2 results/modelB_agg.json \
        --out-json results/stat_test.json
"""
import argparse
import json
import os
from scipy.stats import ttest_rel, wilcoxon  # scipy 설치 필요

def main():
    parser = argparse.ArgumentParser(description='모델 간 통계 검정 수행')
    parser.add_argument('--json1', required=True, help='모델 A의 집계 결과 JSON 경로')  # 수정필요: 파일명 패턴 확인
    parser.add_argument('--json2', required=True, help='모델 B의 집계 결과 JSON 경로')  # 수정필요
    parser.add_argument('--out-json', required=True, help='통계 검정 결과 저장 JSON 경로')
    args = parser.parse_args()

    # JSON 로드
    with open(args.json1, 'r') as f1:
        data1 = json.load(f1)
    with open(args.json2, 'r') as f2:
        data2 = json.load(f2)

    dist1 = data1.get('dist')
    dist2 = data2.get('dist')
    if dist1 is None or dist2 is None:
        raise ValueError('입력 JSON에 "dist" 키가 없습니다.')
    if len(dist1) != len(dist2):
        raise ValueError('두 분포 길이가 다릅니다: {} vs {}'.format(len(dist1), len(dist2)))

    # paired t-test
    t_stat, p_t = ttest_rel(dist1, dist2)
    # Wilcoxon signed-rank test
    w_stat, p_w = wilcoxon(dist1, dist2)

    # 결과 출력
    print('Paired t-test: t = {:.4f}, p-value = {:.4e}'.format(t_stat, p_t))
    print('Wilcoxon signed-rank: W = {:.4f}, p-value = {:.4e}'.format(w_stat, p_w))

    # JSON 저장
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    result = {
        't_test': {'t_stat': t_stat, 'p_value': p_t},
        'wilcoxon': {'w_stat': w_stat, 'p_value': p_w}
    }
    with open(args.out_json, 'w') as out_f:
        json.dump(result, out_f, indent=2)
    print(f"Statistical test results saved to {args.out_json}")

if __name__ == '__main__':
    main()
