import os
import re
import pandas as pd

# ——————————————————————————————
# 설정: score 디렉토리 경로 & 하위 디렉토리 목록
# ——————————————————————————————
BASE_DIR = "./score"
SUBDIRS = [
    "OPERA_finetune_model_final",
    "OPERA_pretrain_model_final",
    "ToxCast_finetune_model_final",
    "ToxCast_pretrain_model_final",
]

# 인식할 모델 이름 목록 (길이 내림차순 정렬: 긴 이름부터 매칭)
KNOWN_MODELS = sorted(
    ["GIN_GCN", "GIN_GAT", "GCN_GAT", "GIN", "GCN", "GAT"],
    key=lambda x: -len(x)
)

records = []

for sd in SUBDIRS:
    dpath = os.path.join(BASE_DIR, sd)
    if not os.path.isdir(dpath):
        continue

    for fname in os.listdir(dpath):
        if not fname.endswith(".out"):
            continue

        core = fname[:-4]  # ".out" 제거

        # — model_type 파싱 (suffix가 KNOWN_MODELS 중 하나인지 확인)
        model_type = None
        prefix = core
        for mt in KNOWN_MODELS:
            suffix = "_" + mt
            if core.endswith(suffix):
                model_type = mt
                prefix = core[:-len(suffix)]
                break
        if model_type is None:
            # 알 수 없는 패턴이면 건너뜀
            continue

        # — source, target 파싱
        if "pretrain" in sd.lower():
            # pretrain 디렉토리: source[_…]&&… 가 있어도 && 전까지만
            src = prefix.split("&&", 1)[0]
            tgt = ""
        else:
            # finetune/target-only: "source&&target" 구조
            if "&&" in prefix:
                src, tgt = prefix.split("&&", 1)
            else:
                src = prefix
                tgt = ""

        # — 파일 내용 읽기
        full = os.path.join(dpath, fname)
        with open(full, "r") as f:
            lines = f.readlines()
        text = "".join(lines)

        # F1 점수 초기화
        f1_ft = None
        f1_to = None

        if "pretrain" in sd.lower():
            # Pretraining completed. Test F1: 0.xxx
            m = re.search(r"Pretraining completed\.\s*Test F1:\s*([0-9.]+)", text)
            if m:
                f1_to = float(m.group(1))
        else:
            # [3.1/4] 이전 줄에서 Test F1 추출
            for i, line in enumerate(lines):
                if line.strip().startswith("[3.1/4]"):
                    if i > 0:
                        prev = lines[i-1]
                        m_ft = re.search(r"Test F1:\s*([0-9.]+)", prev)
                        if m_ft:
                            f1_ft = float(m_ft.group(1))
                    break
            # fallback: “Finetuning completed. Test F1: …”
            if f1_ft is None:
                m1 = re.search(r"Finetuning completed\. Test F1:\s*([0-9.]+)", text)
                if m1:
                    f1_ft = float(m1.group(1))
            # Target-only TEST performance
            m2 = re.search(
                r"=== Target-only TEST performance ===\s*Test F1:\s*([0-9.]+)", text
            )
            if m2:
                f1_to = float(m2.group(1))

        records.append({
            "source": src,
            "target": tgt,
            "model_type": model_type,
            "F1_finetune": f1_ft,
            "F1_targetonly": f1_to,
        })

# ——————————————————————————————
# DataFrame 생성 및 diff 계산
# ——————————————————————————————
df = pd.DataFrame(records, columns=[
    "source", "target", "model_type", "F1_finetune", "F1_targetonly"
])

# diff = F1_finetune - F1_targetonly (target 있는 경우만)
df["diff"] = df.apply(
    lambda r: (r.F1_finetune - r.F1_targetonly)
    if pd.notna(r.F1_finetune) and pd.notna(r.F1_targetonly) and r.target
    else None,
    axis=1
)

# ——————————————————————————————
# 엑셀로 저장
# ——————————————————————————————
output_path = "./visual/performance_summary.xlsx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_excel(output_path, index=False)

print(f"✅ 저장 완료: {output_path}")