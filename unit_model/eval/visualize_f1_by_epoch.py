import os
import re
import matplotlib.pyplot as plt

# 디렉토리 설정
LOG_DIR = "../score/ToxCast_finetune_model_final"
SAVE_DIR = "../visual/f1/epoch_curve_joint"
os.makedirs(SAVE_DIR, exist_ok=True)

# 정규표현식
finetune_epoch_pattern = re.compile(r"\[Fine-tune\] Epoch (\d+)/\d+.*?Val F1: ([0-9.]+)")
targetonly_epoch_pattern = re.compile(r"\[Target-only\] Epoch (\d+)/\d+.*?Val F1: ([0-9.]+)")
finetune_test_pattern = re.compile(r"Finetuning completed.*?Test F1: ([0-9.]+)")
targetonly_test_pattern = re.compile(r"Target-only TEST performance[\s\S]*?Test F1: ([0-9.]+)")

# 로그 파일 탐색
log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".out")]

for fname in sorted(log_files):
    fpath = os.path.join(LOG_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 에폭별 F1 점수 수집
    finetune_f1s = []
    target_f1s = []
    content = "".join(lines)

    for line in lines:
        if m := finetune_epoch_pattern.search(line):
            finetune_f1s.append((int(m.group(1)), float(m.group(2))))
        elif m := targetonly_epoch_pattern.search(line):
            target_f1s.append((int(m.group(1)), float(m.group(2))))

    # 테스트 F1
    ft_test = ft_match = finetune_test_pattern.search(content)
    tgt_test = tgt_match = targetonly_test_pattern.search(content)
    ft_test_f1 = float(ft_match.group(1)) if ft_match else None
    tgt_test_f1 = float(tgt_match.group(1)) if tgt_match else None

    # 둘 다 있어야 시각화
    if not finetune_f1s or not target_f1s:
        continue

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    ft_epochs, ft_scores = zip(*finetune_f1s)
    tg_epochs, tg_scores = zip(*target_f1s)

    plt.plot(ft_epochs, ft_scores, label="Finetuned (Val)", color="blue", marker="o")
    plt.plot(tg_epochs, tg_scores, label="Target-only (Val)", color="red", marker="o")

    # Test F1 수평선 추가
    if ft_test_f1 is not None:
        plt.axhline(y=ft_test_f1, color="blue", linestyle="--", alpha=0.6, label=f"Finetuned Test F1 = {ft_test_f1:.4f}")
    if tgt_test_f1 is not None:
        plt.axhline(y=tgt_test_f1, color="red", linestyle="--", alpha=0.6, label=f"Target-only Test F1 = {tgt_test_f1:.4f}")

    plt.title(f"F1 Comparison\n{fname.replace('.out', '')}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, fname.replace(".out", "_compare.png"))
    plt.savefig(save_path, dpi=300)
    plt.close()
