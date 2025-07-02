import os
import re
import matplotlib.pyplot as plt

# 로그 루트 디렉토리
LOG_ROOT = "../log/20250602/2054(pretraining 100 epoch)"
FIG_SAVE_ROOT = "../visual/f1/pre"

# 로그 파일 필터링 (pretraining 로그만)
log_files = [f for f in os.listdir(LOG_ROOT) if f.startswith("pre_") and f.endswith(".out")]

for log_file in log_files:
    log_path = os.path.join(LOG_ROOT, log_file)

    # assay name 및 모델 이름 추출
    basename = log_file.replace("pre_", "").replace(".out", "")
    parts = basename.split("_")
    model_type = "_".join(parts[-2:])
    assay_name = "_".join(parts[:-2])

    # F1 점수 파싱
    epoch_f1s = []
    test_f1 = None
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/\d+.*Val F1: ([0-9.]+)", line)
            if m:
                epoch = int(m.group(1))
                f1 = float(m.group(2))
                epoch_f1s.append((epoch, f1))
            if "Test F1" in line:
                t = re.search(r"Test F1: ([0-9.]+)", line)
                if t:
                    test_f1 = float(t.group(1))

    if not epoch_f1s:
        print(f"⚠️ F1 점수 없음: {log_file}")
        continue

    # 그래프 생성
    epochs, f1s = zip(*epoch_f1s)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, f1s, label="Validation F1", marker='o')

    if test_f1 is not None:
        plt.axhline(y=test_f1, color='red', linestyle='--', label=f"Test F1 = {test_f1:.4f}")

    plt.title(f"Pretraining F1 Curve\n{assay_name} ({model_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1 Score")
    plt.grid(True)
    plt.legend()

    # 저장 디렉토리 및 경로
    save_dir = os.path.join(FIG_SAVE_ROOT, model_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{assay_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 저장됨: {save_path}")
