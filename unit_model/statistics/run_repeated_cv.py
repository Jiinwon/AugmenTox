#!/usr/bin/env python3
"""
run_repeated_cv.py

Repeated K-Fold Cross-Validation 실행 스크립트

Usage:
    python run_repeated_cv.py \
        --data-path data.pkl \
        --val-idx-dir ./val_indices \
        --repeats 10 --folds 5 \
        --batch_size 32 --device cuda \
        --out-csv results/gnn_scores.csv

설명:
    - 상위 디렉토리의 config 모듈을 사용하기 위해 PYTHONPATH 조정
    - evaluate_model.py 스크립트가 --val_idx와 --out 인자만 받음
"""
import sys
import os
import subprocess
# 상위 디렉토리에서 config, data, train 모듈을 임포트할 수 있도록 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 수정필요: 실제 프로젝트 구조에 맞춰 경로 조정

import argparse
import json
import csv
from sklearn.model_selection import RepeatedKFold
import config.config as cfg  # 수정필요: config 모듈 경로 확인
from data.load_data import load_data  # 수정필요: 데이터 로드 함수 경로 확인


def main():
    parser = argparse.ArgumentParser(description='Repeated K-Fold Cross-Validation')
    parser.add_argument('--data-path', required=True, help='전체 데이터 파일 경로 (예: data.pkl)')  # 수정필요
    parser.add_argument('--val-idx-dir', required=True, help='검증 인덱스 및 결과 저장 디렉토리')
    parser.add_argument('--repeats', type=int, default=10, help='반복 횟수')
    parser.add_argument('--folds', type=int, default=5, help='폴드 수')
    parser.add_argument('--batch_size', type=int, default=getattr(cfg, 'BATCH_SIZE', 32), help='배치 크기')
    parser.add_argument('--device', default='cuda', help='연산 디바이스 (cuda 또는 cpu)')
    parser.add_argument('--model-path', required=True, help='평가할 모델 파일 경로')
    parser.add_argument('--out-csv', required=True, help='출력 CSV 파일 경로')
    args = parser.parse_args()

    # 전체 검증 데이터 로드
    _, _, full_test = load_data(
        args.data_path,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        random_seed=cfg.RANDOM_SEED
    )
    n_samples = len(full_test)

    # Repeated K-Fold 분할
    rkf = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)
    splits = list(rkf.split(range(n_samples)))

    # 디렉토리 준비
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(args.val_idx_dir, exist_ok=True)

    # CSV 헤더 작성
    with open(args.out_csv, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['rep', 'fold', 'f1', 'roc_auc', 'pr_auc'])

    idx = 0
    # 반복 실행
    for rep in range(1, args.repeats + 1):
        for fold in range(1, args.folds + 1):
            _, val_indices = splits[idx]
            idx += 1

            # 검증 인덱스 JSON 저장
            val_file = os.path.join(args.val_idx_dir, f'val_idx_rep{rep}_fold{fold}.json')
            with open(val_file, 'w') as vf:
                json.dump(val_indices, vf)

            # evaluate_model.py 호출
            metrics_file = os.path.join(args.val_idx_dir, f'metrics_rep{rep}_fold{fold}.json')
            eval_script = os.path.join(os.path.dirname(__file__), 'evaluate_model.py')
            cmd = [
                'python3', eval_script,
                '--val_idx', val_file,
                '--batch_size', str(args.batch_size),
                '--device', args.device,
                '--model-path', args.model_path,
                '--out', metrics_file
            ]
            subprocess.run(cmd, check=True)  # 수정필요: evaluate_model.py 경로 확인

            # 결과 로드 및 CSV에 기록
            with open(metrics_file) as mf:
                m = json.load(mf)
            with open(args.out_csv, 'a', newline='') as out_f:
                writer = csv.writer(out_f)
                writer.writerow([rep, fold, m['f1'], m.get('roc_auc'), m.get('pr_auc')])
            print(f'Rep {rep} Fold {fold} → F1: {m["f1"]:.4f}')

    print(f'Results saved to {args.out_csv}')


if __name__ == '__main__':
    main()
