import os
import re
import matplotlib.pyplot as plt

# 저장 경로 설정
FIG_SAVE_ROOT = "../visual/f1/data_by_data"
os.makedirs(FIG_SAVE_ROOT, exist_ok=True)
save_path = os.path.join(FIG_SAVE_ROOT, "f1_gain_comparison_boxplot.png")

# 실험 로그 디렉토리
DIRS = {
    "ToxCast": "../score/ToxCast_finetune_model_final",
    "OPERA": "../score/OPERA_finetune_model_final"
}

# 정규표현식
finetuned_f1_pattern = re.compile(r"Fine-tuned TEST performance[\s\S]*?Test F1: ([0-9.]+)")
completed_f1_pattern = re.compile(r"Finetuning completed[\s\S]*?Test F1: ([0-9.]+)")
targetonly_f1_pattern = re.compile(r"Target-only TEST performance[\s\S]*?Test F1: ([0-9.]+)")

# 결과 저장
f1_gains_by_dataset = {"ToxCast": [], "OPERA": []}

for label, directory in DIRS.items():
    for filename in os.listdir(directory):
        if not filename.endswith(".out"):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # F1 점수 추출
        ft_match = finetuned_f1_pattern.search(content) or completed_f1_pattern.search(content)
        tgt_match = targetonly_f1_pattern.search(content)

        if not ft_match or not tgt_match:
            print(f"[무시됨] F1 추출 실패: {filename}")
            continue

        ft_f1 = float(ft_match.group(1))
        tgt_f1 = float(tgt_match.group(1))
        gain = ft_f1 - tgt_f1

        f1_gains_by_dataset[label].append(gain)

        print(f"[{label}] Fine-tuned: {ft_f1:.4f}, Target-only: {tgt_f1:.4f}, Gain: {gain:.4f}")

# 그래프
plt.figure(figsize=(8, 6))
plt.boxplot(
    [f1_gains_by_dataset["ToxCast"], f1_gains_by_dataset["OPERA"]],
    labels=["ToxCast", "OPERA"]
)
plt.title("F1 Score Gain Distribution: ToxCast vs OPERA")
plt.ylabel("F1 Score Improvement (Fine-tuned - Target-only)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ 그래프 저장 완료: {save_path}")
