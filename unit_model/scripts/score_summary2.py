#!/usr/bin/env python3
# scripts/score_summary.py

import os
import re
import glob
import sys
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# 경로 설정: 부모 디렉토리(unit_model)를 PYTHONPATH에 추가
CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(BASE_DIR)

import config.config as cfg  # config/config.py에 정의된 SOURCE_NAMES, TARGET_NAMES 등
# ───────────────────────────────────────────────────────────────────────────────

# 설정
LOG_ROOT    = os.path.join(BASE_DIR, "log", "20250530", "0124")
OUTPUT_XLSX = "f1_summary3.xlsx"
MODEL_TYPES = ["GIN"]
TARGET_SOURCE = "Supplemental_Material_2_TrainingSet"

# 1) .out 파일 목록 수집
pattern = os.path.join(LOG_ROOT, "*.out")
out_paths = glob.glob(pattern)

# 2) 파일명 및 F1 파싱 정규식
fn_re     = re.compile(r"([^/\\]+?)&&([^/\\]+?)_([A-Za-z0-9]+)\.out$")
pre_f1_re = re.compile(r"Pretraining completed\. Test F1:\s*([0-9]*\.?[0-9]+)(?=,|$)")
ft_f1_re  = re.compile(r"Finetuning completed\. Test F1:\s*([0-9]*\.?[0-9]+)(?=,|$)")
to_f1_re  = re.compile(r"=== Target-only TEST performance ===[\s\S]*?Test F1:\s*([0-9]*\.?[0-9]+)(?=,|$)")

# 3) 결과 저장용 딕셔너리
pre_results = {m: {} for m in MODEL_TYPES}
ft_results  = {m: {} for m in MODEL_TYPES}
to_results  = {m: {} for m in MODEL_TYPES}

# 4) 로그 파일 파싱
for path in out_paths:
    fn = os.path.basename(path)
    m = fn_re.match(fn)
    if not m:
        continue
    src, tgt, model = m.groups()
    if model not in MODEL_TYPES:
        continue
    if src != TARGET_SOURCE:
        continue

    txt = open(path, encoding="utf-8", errors="ignore").read()

    # Pre-training F1
    pre_matches = pre_f1_re.findall(txt)
    if pre_matches:
        f1_pre = float(pre_matches[-1])
        pre_results[model][src] = f1_pre

    # Fine-tuning F1
    ft_matches = ft_f1_re.findall(txt)
    if ft_matches:
        f1_ft = float(ft_matches[-1])
        ft_results[model].setdefault(src, {})[tgt] = f1_ft

    # Target-only F1
    tm = to_f1_re.search(txt)
    if tm:
        f1_to = float(tm.group(1))
        to_results[model][tgt] = f1_to

# 5) 엑셀로 저장 (해당 source만 포함)
with pd.ExcelWriter(OUTPUT_XLSX) as writer:
    for model in MODEL_TYPES:
        idx = [TARGET_SOURCE, "Target_only"]
        cols = ["Pre-training"] + cfg.TARGET_NAMES
        df = pd.DataFrame(index=idx, columns=cols, dtype=float)

        # Pre-training
        df.at[TARGET_SOURCE, "Pre-training"] = pre_results.get(model, {}).get(TARGET_SOURCE, pd.NA)
        df.at["Target_only", "Pre-training"] = pd.NA

        # Fine-tuning
        for tgt in cfg.TARGET_NAMES:
            df.at[TARGET_SOURCE, tgt] = ft_results.get(model, {}).get(TARGET_SOURCE, {}).get(tgt, pd.NA)

        # Target-only
        for tgt in cfg.TARGET_NAMES:
            df.at["Target_only", tgt] = to_results.get(model, {}).get(tgt, pd.NA)

        # 소수점 4자리 문자열로 포맷 후 저장
        df_formatted = df.applymap(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        df_formatted.to_excel(writer, sheet_name=model)

print(f"✅ F1 요약 엑셀 생성 완료: {OUTPUT_XLSX}")
