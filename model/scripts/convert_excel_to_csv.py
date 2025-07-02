#!/usr/bin/env python3
import os
import pandas as pd
import config.config as cfg

def main():
    # 1. 전체 엑셀 파일 읽기
    excel_path = cfg.ENTIRE_DATA_PATH
    if excel_path.endswith(('.xls','xlsx')):
        df = pd.read_excel(excel_path)
    else:
        df = pd.read_csv(excel_path)

    # 2. 공통: SMILES 열이 반드시 있어야 함
    if 'SMILES' not in df.columns:
        raise KeyError("Excel에 'SMILES' 컬럼이 없습니다!")
    df = df.dropna(subset=['SMILES'])

    # 3. SOURCE_NAMES 기준 CSV 생성
    for src in cfg.SOURCE_NAMES:
        if src not in df.columns:
            print(f"[WARN] '{src}' 컬럼이 없어 SKIP")
            continue
        df_src = df[['SMILES', src]].dropna()
        df_src.columns = ['smiles', 'label']
        out_dir = os.path.join("data", "sample")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{src}.csv")
        df_src.to_csv(out_path, index=False)
        print(f"Saved pretrain CSV for {src}: {out_path}")

    # 4. TARGET_NAMES 기준 CSV 생성
    for tgt in cfg.TARGET_NAMES:
        if tgt not in df.columns:
            print(f"[WARN] '{tgt}' 컬럼이 없어 SKIP")
            continue
        df_tgt = df[['SMILES', tgt]].dropna()
        df_tgt.columns = ['smiles', 'label']
        out_dir = os.path.join("data", "sample")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{tgt}.csv")
        df_tgt.to_csv(out_path, index=False)
        print(f"Saved finetune CSV for {tgt}: {out_path}")

if __name__ == "__main__":
    main()