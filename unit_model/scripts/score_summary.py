#!/usr/bin/env python3
# scripts/score_summary.py

import os
import re
import glob
import pandas as pd
import config.config as cfg

# ───────────────────────────────────────────────────────────────────────────────
# 설정
LOG_ROOT    = "./log/20250514"                   # 로그가 저장된 최상위 디렉토리
OUTPUT_XLSX = "f1_summary.xlsx"                  # 출력할 엑셀 파일명
MODEL_TYPES = ["GIN", "GCN", "GAT", "GIN_GCN", "GCN_GAT", "GIN_GAT"]              # 돌린 모델 타입 목록
# ───────────────────────────────────────────────────────────────────────────────

# 1) .out 파일 목록
pattern = os.path.join(LOG_ROOT, "*", "*.out")
out_paths = glob.glob(pattern)

# 2) 파일명·F1 파싱용 정규식
fn_re         = re.compile(r"([^/\\]+?)&&([^/\\]+?)_([A-Za-z0-9]+)\.out$")
pre_f1_re     = re.compile(r"Pretraining completed\. Test F1:\s*([0-9]*\.?[0-9]+)")
ft_f1_re      = re.compile(r"Finetuning completed\. Test F1:\s*([0-9]*\.?[0-9]+)")
to_f1_re      = re.compile(
    r"=== Target-only TEST performance ===[\s\S]*?Test F1:\s*([0-9]*\.?[0-9]+)"
)

# 3) 파싱 결과 저장용
pre_results  = {m: {} for m in MODEL_TYPES}
ft_results   = {m: {} for m in MODEL_TYPES}
to_results   = {m: {} for m in MODEL_TYPES}

for path in out_paths:
    fn = os.path.basename(path)
    m = fn_re.match(fn)
    if not m:
        continue
    src, tgt, model = m.groups()
    if model not in MODEL_TYPES:
        continue

    txt = open(path, encoding="utf-8", errors="ignore").read()

    # 3.1) Pre-training F1
    pm = pre_f1_re.search(txt)
    if pm:
        f1_pre = float(pm.group(1))
        pre_results[model][src] = f1_pre

    # 3.2) Fine-tuning F1
    fm = ft_f1_re.search(txt)
    if fm:
        f1_ft = float(fm.group(1))
        ft_results[model].setdefault(src, {})[tgt] = f1_ft

    # 3.3) Target-only F1
    tm = to_f1_re.search(txt)
    if tm:
        f1_to = float(tm.group(1))
        to_results[model][tgt] = f1_to

# 4) 엑셀로 쓰기
with pd.ExcelWriter(OUTPUT_XLSX) as writer:
    for model in MODEL_TYPES:
        # 행: SOURCE_NAMES + ["Target_only"]
        idx = list(cfg.SOURCE_NAMES) + ["Target_only"]
        # 열: ["Pre-training"] + TARGET_NAMES
        cols = ["Pre-training"] + cfg.TARGET_NAMES

        df = pd.DataFrame(index=idx, columns=cols, dtype=float)

        # 4.1) Pre-training 열 채우기
        for src in cfg.SOURCE_NAMES:
            df.at[src, "Pre-training"] = pre_results.get(model, {}).get(src, pd.NA)
        df.at["Target_only", "Pre-training"] = pd.NA

        # 4.2) Fine-tuning 결과 채우기
        for src in cfg.SOURCE_NAMES:
            for tgt in cfg.TARGET_NAMES:
                df.at[src, tgt] = ft_results.get(model, {}).get(src, {}).get(tgt, pd.NA)

        # 4.3) Target-only 행 채우기
        for tgt in cfg.TARGET_NAMES:
            df.at["Target_only", tgt] = to_results.get(model, {}).get(tgt, pd.NA)

        # 시트별로 저장
        df.to_excel(writer, sheet_name=model)

print(f"✅ F1 요약 엑셀 생성 완료: {OUTPUT_XLSX}")