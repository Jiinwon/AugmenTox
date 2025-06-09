#!/usr/bin/env python3
"""
evaluate_model.py

파인튜닝된 모델(.pth)과 검증 인덱스(JSON)를 이용해
검증 데이터에서 F1, ROC-AUC, PR-AUC를 계산한 후 JSON으로 저장합니다.

Usage:
    python evaluate_model.py --val_idx val_idx_rep1_fold1.json --out metrics_rep1_fold1.json

환경 설정:
    - 상위 디렉토리의 config 모듈 불러오기 위해 PYTHONPATH 조정
    - 환경 변수 SOURCE_NAME, TARGET_NAME, MODEL_TYPE이 설정되어 있어야 함
"""

import sys, os
# 상위 디렉토리에서 config, data, train 모듈을 불러오기 위한 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import torch
from torch_geometric.loader import DataLoader

import config.config as cfg      # 수정필요: 프로젝트 구조에 맞게 조정했음을 확인하세요
from data.load_data import load_data    # 수정필요: data/load_data.py 위치 확인
from train.utils import load_model, evaluate as calc_metrics  # 수정필요: train/utils.py 위치 확인

def main():
    parser = argparse.ArgumentParser(description='모델 평가 스크립트')
    parser.add_argument(
        '--val_idx', required=True,
        help='검증 인덱스 JSON 파일 경로'
    )
    parser.add_argument(
        '--batch_size', type=int, default=cfg.BATCH_SIZE,
        help='배치 크기 (기본: config.BATCH_SIZE)'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='연산 디바이스 (cuda 또는 cpu)'
    )
    parser.add_argument(
        '--out', required=True,
        help='출력 JSON 파일 경로'
    )
    args = parser.parse_args()

    # 전체 테스트 데이터 로드
    _, _, full_test = load_data(
        cfg.TARGET_DATA_PATH,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        random_seed=cfg.RANDOM_SEED
    )

    # 검증 인덱스 로드
    with open(args.val_idx, 'r') as f:
        val_indices = json.load(f)

    # 검증 서브셋 구성
    val_subset = [full_test[i] for i in val_indices]
    loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # 모델 로드
    model_path = cfg.FINETUNED_MODEL_PATH
    mt = cfg.MODEL_TYPE.upper()
    input_dim = full_test[0].x.shape[1]
    output_dim = cfg.NUM_CLASSES

    if mt == 'GIN':
        from models.gin import GINNet as Net
        model = Net(input_dim, cfg.HIDDEN_DIM, output_dim,
                    num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif mt == 'GCN':
        from models.gcn import GCNNet as Net
        model = Net(input_dim, cfg.HIDDEN_DIM, output_dim,
                    num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif mt == 'GAT':
        from models.gat import GATNet as Net
        model = Net(input_dim, cfg.HIDDEN_DIM, output_dim,
                    num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
    elif mt == 'GIN_GCN':
        from models.gin_gcn import GIN_GCN_Hybrid as Net
        model = Net(input_dim, cfg.HIDDEN_DIM, output_dim,
                    num_layers=cfg.NUM_LAYERS, dropout=cfg.DROPOUT)
    elif mt == 'GIN_GAT':
        from models.gin_gat import GIN_GAT_Hybrid as Net
        model = Net(input_dim, cfg.HIDDEN_DIM, output_dim,
                    num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
    elif mt == 'GCN_GAT':
        from models.gcn_gat import GCN_GAT_Hybrid as Net
        model = Net(input_dim, cfg.HIDDEN_DIM, output_dim,
                    num_layers=cfg.NUM_LAYERS, heads=cfg.NUM_HEADS, dropout=cfg.DROPOUT)
    else:
        raise ValueError(f"지원하지 않는 MODEL_TYPE: {cfg.MODEL_TYPE}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    load_model(model, model_path, device)

    # 평가 수행
    metrics = calc_metrics(model, loader, device)

    # 결과 저장
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'f1': metrics['f1'],
            'roc_auc': metrics.get('roc_auc'),
            'pr_auc': metrics.get('pr_auc')
        }, f, indent=2)

    print(f"Evaluation metrics saved to {args.out}")

if __name__ == '__main__':
    main()