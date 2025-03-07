import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# 3. 원하는 프로젝트와 엔터티의 실행 목록 가져오기
runs = api.runs("hails/condg_mlp")

# 4. 특정 run 찾기
# run_names = [
#     "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti",
#     "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use",
#     "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11",
#     "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28",
#     "unst_cnn_cifar10_lth_real10",
#     "st_cnn_cifar10_lth_real10",
# ]
# colors = ["blue", "green", "red", "orange", "black", "black"]
#
# legend_labels = {
#     "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti": "CondGNet (Ours)",
#     "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use": "CondNet",
#     "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11": "Runtime Activation Magnitude",
#     "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28": "Runtime Weight Magnitude",
#     "unst_cnn_cifar10_lth_real10": "Unstructured LTH",
#     "st_cnn_cifar10_lth_real10": "Structured LTH"
#     # "st_cnn_cifar10_lth_real10": "Structured Pruning (LTH)"
# }

run_names = [
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11",
    "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50",
    "unst_mlp_mnist_lth_real10",
    "st_mlp_mnist_lth_real10",
]
colors = ["blue", "green", "red", "orange", "black", "black"]

legend_labels = {
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-": "CondGNet (Ours)",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use": "CondNet",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11": "Runtime Activation Magnitude",
    "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50": "Runtime Weight Magnitude",
    "unst_mlp_mnist_lth_real10": "Unstructured LTH",
    "st_mlp_mnist_lth_real10": "Structured LTH"
    # "st_cnn_cifar10_lth_real10": "Structured Pruning (LTH)"
}

found_runs = {}
for run in runs:
    if run.name in run_names:
        found_runs[run.name] = run
    if len(found_runs) == len(run_names):
        break

# 5. 해당 run이 없는 경우 에러 처리
if len(found_runs) != len(run_names):
    raise ValueError("일부 지정된 run을 찾을 수 없습니다.")

# 6. 각 run의 로그 데이터 추출
dataframes = {}
# cnn is 8, mlp is 5
for name, run in found_runs.items():
    if name in ["st_mlp_mnist_lth_real10", "unst_mlp_mnist_lth_real10"]:
        history = run.scan_history(keys=["Pruning Iteration 5/test/epoch_tau"])
        epoch_acc_values = [row["Pruning Iteration 5/test/epoch_tau"] for row in history if
                            "Pruning Iteration 5/test/epoch_tau" in row]
    else:
        history = run.scan_history(keys=["test/epoch_tau"])
        epoch_acc_values = [row["test/epoch_tau"] for row in history if "test/epoch_tau" in row]

    epochs = list(range(1, len(epoch_acc_values) + 1))

    if name in ["st_mlp_mnist_lth_real10", "unst_mlp_mnist_lth_real10"]:
        # 25~30 epoch의 mean 값 계산
        mean_tau = np.mean(epoch_acc_values[27:30])

        # 31~200 epoch을 mean_tau로 연장
        extended_epochs = list(range(31, 201))
        extended_tau = [mean_tau] * len(extended_epochs)

        # 기존 데이터와 점근선 결합
        epochs.extend(extended_epochs)
        epoch_acc_values.extend(extended_tau)

    dataframes[name] = pd.DataFrame({
        "Epoch": epochs,
        "Tau": epoch_acc_values
    })

# 7. 플롯 생성
# plt.figure(figsize=(10, 6))
plt.figure(figsize=(2.6, 2))
for idx, name in enumerate(run_names):
    df = dataframes[name]
    if name == "st_mlp_mnist_lth_real10":
        # 점선 (dot) → 대쉬 (--)
        plt.plot(df["Epoch"][:30], df["Tau"][:30], color=colors[idx], label=legend_labels[name], linestyle=':')
        plt.plot(df["Epoch"][30:], df["Tau"][30:], color=colors[idx], linestyle='--')

    elif name == "unst_mlp_mnist_lth_real10":
        # 실선 (-) → 대쉬 (--)
        plt.plot(df["Epoch"][:30], df["Tau"][:30], color=colors[idx], label=legend_labels[name], linestyle='-')
        plt.plot(df["Epoch"][30:], df["Tau"][30:], color=colors[idx], linestyle='--')
    else:
        plt.plot(df["Epoch"], df["Tau"], color=colors[idx], label=legend_labels[name], linestyle='-')


plt.xlabel("Epoch", fontsize=9)
plt.ylabel(r"$\tau$", fontsize=9)
plt.title("")
# plt.legend(loc="lower right", fontsize=9, framealpha=0.8, ncol=2)
# plt.legend(loc="lower right", fontsize=9, framealpha=0.8)
plt.grid(False)
plt.ylim(0, 1)
# plt.yticks(np.arange(0.0, 1.1, 0.1))
y_ticks = np.arange(0, 1.1, 0.2)
plt.yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks])
plt.xlim(1, 200)
plt.xticks(fontsize=9)
plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 8. 플롯 출력
plt.savefig("mlp_all_tau.pdf", format="pdf", bbox_inches = 'tight')
