# import os
# import re
# import matplotlib.pyplot as plt

# # 디렉토리 설정
# base_dir = "../score/ToxCast_finetune_model_final"
# FIG_SAVE_ROOT = "../visual/f1/model_by_model"
# os.makedirs(FIG_SAVE_ROOT, exist_ok=True)
# save_path = os.path.join(FIG_SAVE_ROOT, "avg_f1_gain_per_model.png")

# # 모델 키 정의
# model_keys = ["GIN", "GCN", "GAT", "GIN_GCN", "GIN_GAT", "GCN_GAT"]
# f1_improvements = {key: [] for key in model_keys}

# # 정규표현식 (줄바꿈 포함 대응)
# finetuned_f1_pattern = re.compile(r"Fine-tuned TEST performance[\s\S]*?Test F1: ([0-9.]+)")
# completed_f1_pattern = re.compile(r"Finetuning completed[\s\S]*?Test F1: ([0-9.]+)")
# targetonly_f1_pattern = re.compile(r"Target-only TEST performance[\s\S]*?Test F1: ([0-9.]+)")

# # 파일 순회
# for filename in os.listdir(base_dir):
#     if not filename.endswith(".out"):
#         continue

#     filepath = os.path.join(base_dir, filename)
#     with open(filepath, "r", encoding="utf-8") as f:
#         content = f.read()

#     # 모델명 정확 추출 로직
#     model_name = None
#     try:
#         suffix = filename.split("&&")[1].replace(".out", "").replace(".err", "")
#         parts = suffix.split("_")
#         for i in range(3, 0, -1):  # 긴 조합부터
#             candidate = "_".join(parts[-i:])
#             if candidate.upper() in model_keys:
#                 model_name = candidate.upper()
#                 break
#     except Exception as e:
#         print(f"[경고] 모델명 파싱 실패: {filename} → {e}")
#         continue

#     if not model_name:
#         print(f"[무시됨] 모델명 추출 실패: {filename}")
#         continue

#     # F1 점수 추출
#     ft_match = finetuned_f1_pattern.search(content) or completed_f1_pattern.search(content)
#     tgt_match = targetonly_f1_pattern.search(content)

#     if not ft_match or not tgt_match:
#         print(f"❌ F1 추출 실패: {filename}")
#         continue

#     ft_f1 = float(ft_match.group(1))
#     tgt_f1 = float(tgt_match.group(1))
#     delta_f1 = ft_f1 - tgt_f1
#     f1_improvements[model_name].append(delta_f1)

#     print(f"[{model_name}] Fine-tuned F1: {ft_f1:.4f}, Target-only F1: {tgt_f1:.4f}, Gain: {delta_f1:.4f}")

# # 평균 계산
# average_gains = {
#     model: sum(vals) / len(vals) if vals else 0.0
#     for model, vals in f1_improvements.items()
# }

# # 바그래프 저장
# plt.figure(figsize=(10, 6))
# plt.bar(average_gains.keys(), average_gains.values(), color='skyblue')
# plt.ylabel("Average F1 Score Improvement")
# plt.title("Average F1 Score Gain after Finetuning per Model")
# plt.xticks(rotation=45)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# plt.close()

# print(f"✅ 그래프 저장 완료: {save_path}")


import os
import re
import matplotlib.pyplot as plt

# 디렉토리 설정
base_dir = "../score/ToxCast_finetune_model_final"
FIG_SAVE_ROOT = "../visual/f1/model_by_model"
os.makedirs(FIG_SAVE_ROOT, exist_ok=True)
save_path = os.path.join(FIG_SAVE_ROOT, "boxplot_f1_gain_per_model.png")

# 모델 키 정의
model_keys = ["GIN", "GCN", "GAT", "GIN_GCN", "GIN_GAT", "GCN_GAT"]
f1_improvements = {key: [] for key in model_keys}

# 정규표현식 (줄바꿈 포함 대응)
finetuned_f1_pattern = re.compile(r"Fine-tuned TEST performance[\s\S]*?Test F1: ([0-9.]+)")
completed_f1_pattern = re.compile(r"Finetuning completed[\s\S]*?Test F1: ([0-9.]+)")
targetonly_f1_pattern = re.compile(r"Target-only TEST performance[\s\S]*?Test F1: ([0-9.]+)")

# 파일 순회
for filename in os.listdir(base_dir):
    if not filename.endswith(".out"):
        continue

    filepath = os.path.join(base_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 모델명 정확 추출 로직
    model_name = None
    try:
        suffix = filename.split("&&")[1].replace(".out", "").replace(".err", "")
        parts = suffix.split("_")
        for i in range(3, 0, -1):
            candidate = "_".join(parts[-i:])
            if candidate.upper() in model_keys:
                model_name = candidate.upper()
                break
    except Exception as e:
        print(f"[경고] 모델명 파싱 실패: {filename} → {e}")
        continue

    if not model_name:
        print(f"[무시됨] 모델명 추출 실패: {filename}")
        continue

    # F1 점수 추출
    ft_match = finetuned_f1_pattern.search(content) or completed_f1_pattern.search(content)
    tgt_match = targetonly_f1_pattern.search(content)

    if not ft_match or not tgt_match:
        print(f"❌ F1 추출 실패: {filename}")
        continue

    ft_f1 = float(ft_match.group(1))
    tgt_f1 = float(tgt_match.group(1))
    delta_f1 = ft_f1 - tgt_f1
    f1_improvements[model_name].append(delta_f1)

    print(f"[{model_name}] Fine-tuned F1: {ft_f1:.4f}, Target-only F1: {tgt_f1:.4f}, Gain: {delta_f1:.4f}")

# Boxplot 그리기
plt.figure(figsize=(10, 6))
data = [f1_improvements[key] for key in model_keys]
plt.boxplot(data, labels=model_keys, patch_artist=True)
plt.ylabel("F1 Score Improvement (Fine-tuned - Target-only)")
plt.title("F1 Score Gain Distribution after Finetuning per Model")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ 박스플롯 그래프 저장 완료: {save_path}")
