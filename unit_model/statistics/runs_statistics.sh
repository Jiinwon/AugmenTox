#!/usr/bin/env bash
# run_all.sh: 1→2→3→4→5를 순차 실행

set -e

# 1. 반복 CV
python run_repeated_cv.py \
  --data-path data.pkl \
  --val-idx-dir val_indices \
  --repeats 10 --folds 5 \
  --batch_size 32 --device cuda \
  --out-csv results/gnn_scores.csv

# 2. 집계
python aggregate_results.py \
  --input-csv results/gnn_scores.csv \
  --out-json results/agg_results.json

# 3. 통계 검정
python stat_test.py \
  --json1 results/agg_results_modelA.json \   # 수정필요: 파일명
  --json2 results/agg_results_modelB.json \   # 수정필요
  --out-json results/stat_test.json

# 4. 시각화
python visualize_results.py \
  --json1 results/agg_results_modelA.json \  # 수정필요
  --json2 results/agg_results_modelB.json \  # 수정필요
  --labels ModelA ModelB \
  --out-dir figures