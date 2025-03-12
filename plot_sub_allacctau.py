import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# 3. 원하는 프로젝트와 엔터티의 실행 목록 가져오기
runs = api.runs("hails/condg_cnn")

# 4. 특정 run 찾기
# run_names = [
#     "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-",
#     "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use",
#     "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11",
#     "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50",
#     "unst_mlp_mnist_lth_real10",
#     "st_mlp_mnist_lth_real10",
# ]
# colors = ["blue", "green", "red", "orange", "black", "black"]
#
# legend_labels = {
#     "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-": "CondGNet (Ours)",
#     "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use": "CondNet",
#     "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11": "Runtime Activation Magnitude",
#     "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50": "Runtime Weight Magnitude",
#     "unst_mlp_mnist_lth_real10": "Unstructured LTH",
#     "st_mlp_mnist_lth_real10": "Structured LTH"
#     # "st_cnn_cifar10_lth_real10": "Structured Pruning (LTH)"
# }

run_names = [
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28",
    "unst_cnn_cifar10_lth_real10",
    "st_cnn_cifar10_lth_real10",
]
colors = ["blue", "green", "red", "orange", "black", "black"]

legend_labels = {
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti": "CondGNet (Ours)",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use": "CondNet",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11": "Runtime Activation Magnitude",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28": "Runtime Weight Magnitude",
    "unst_cnn_cifar10_lth_real10": "Unstructured LTH",
    "st_cnn_cifar10_lth_real10": "Structured LTH"
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
    if name in ["st_cnn_cifar10_lth_real10", "unst_cnn_cifar10_lth_real10"]:
        history = run.scan_history(keys=["Pruning Iteration 8/test/epoch_acc"])
        epoch_acc_values = [row["Pruning Iteration 8/test/epoch_acc"] for row in history if
                            "Pruning Iteration 8/test/epoch_acc" in row]
    else:
        history = run.scan_history(keys=["test/epoch_acc"])
        epoch_acc_values = [row["test/epoch_acc"] for row in history if "test/epoch_acc" in row]

    epochs = list(range(1, len(epoch_acc_values) + 1))

    if name in ["st_cnn_cifar10_lth_real10", "unst_cnn_cifar10_lth_real10"]:
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
        "Accuracy": epoch_acc_values
    })



# 4. 특정 run 찾기
# run_names = [
#     "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-",
#     "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use",
#     "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11",
#     "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50",
#     "unst_mlp_mnist_lth_real10",
#     "st_mlp_mnist_lth_real10",
# ]
# colors = ["blue", "green", "red", "orange", "black", "black"]
#
# legend_labels = {
#     "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-": "CondGNet (Ours)",
#     "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use": "CondNet",
#     "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11": "Runtime Activation Magnitude",
#     "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50": "Runtime Weight Magnitude",
#     "unst_mlp_mnist_lth_real10": "Unstructured LTH",
#     "st_mlp_mnist_lth_real10": "Structured LTH"
#     # "st_cnn_cifar10_lth_real10": "Structured Pruning (LTH)"
# }

run_names = [
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28",
    "unst_cnn_cifar10_lth_real10",
    "st_cnn_cifar10_lth_real10",
]
colors = ["blue", "green", "red", "orange", "black", "black"]

legend_labels = {
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti": "CondGNet (Ours)",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use": "CondNet",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11": "Runtime Activation Magnitude",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28": "Runtime Weight Magnitude",
    "unst_cnn_cifar10_lth_real10": "Unstructured LTH",
    "st_cnn_cifar10_lth_real10": "Structured LTH"
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
dataframes_tau = {}
# cnn is 8, mlp is 5
for name, run in found_runs.items():
    if name in ["st_cnn_cifar10_lth_real10", "unst_cnn_cifar10_lth_real10"]:
        history = run.scan_history(keys=["Pruning Iteration 8/test/epoch_tau"])
        epoch_acc_values = [row["Pruning Iteration 8/test/epoch_tau"] for row in history if
                            "Pruning Iteration 8/test/epoch_tau" in row]
    else:
        history = run.scan_history(keys=["test/epoch_tau"])
        epoch_acc_values = [row["test/epoch_tau"] for row in history if "test/epoch_tau" in row]

    epochs = list(range(1, len(epoch_acc_values) + 1))

    if name in ["st_cnn_cifar10_lth_real10", "unst_cnn_cifar10_lth_real10"]:
        # 25~30 epoch의 mean 값 계산
        mean_tau = np.mean(epoch_acc_values[27:30])

        # 31~200 epoch을 mean_tau로 연장
        extended_epochs = list(range(31, 201))
        extended_tau = [mean_tau] * len(extended_epochs)

        # 기존 데이터와 점근선 결합
        epochs.extend(extended_epochs)
        epoch_acc_values.extend(extended_tau)

    dataframes_tau[name] = pd.DataFrame({
        "Epoch": epochs,
        "Tau": epoch_acc_values
    })

# 7. 플롯 생성
fig, ax = plt.subplots(1,2, figsize=(5.3, 2), width_ratios=[1,1], constrained_layout=True)
fig.tight_layout(pad=1)

for idx, name in enumerate(run_names):
    df = dataframes[name]
    if name == "st_cnn_cifar10_lth_real10":
        # 점선 (dot) → 대쉬 (--)
        ax[0].plot(df["Epoch"][:30], df["Accuracy"][:30], color=colors[idx], label=legend_labels[name], linestyle=':')
        ax[0].plot(df["Epoch"][30:], df["Accuracy"][30:], color=colors[idx], linestyle='--')

    elif name == "unst_cnn_cifar10_lth_real10":
        # 실선 (-) → 대쉬 (--)
        ax[0].plot(df["Epoch"][:30], df["Accuracy"][:30], color=colors[idx], label=legend_labels[name], linestyle='-')
        ax[0].plot(df["Epoch"][30:], df["Accuracy"][30:], color=colors[idx], linestyle='--')
    else:
        ax[0].plot(df["Epoch"], df["Accuracy"], color=colors[idx], label=legend_labels[name], linestyle='-')

# 축 및 제목 설정
ax[0].set_xlabel("Epoch", fontsize=9)
ax[0].set_ylabel("Accuracy", fontsize=9)
ax[0].grid(False)
ax[0].set_ylim(0, 1)
y_ticks = np.arange(0, 1.1, 0.2)
ax[0].set_yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks], fontsize=9)
ax[0].set_xlim(1, 200)
ax[0].tick_params(axis='x', labelsize=9)

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)


for idx, name in enumerate(run_names):
    df = dataframes_tau[name]
    if name == "st_cnn_cifar10_lth_real10":
        # 점선 (dot) → 대쉬 (--)
        ax[1].plot(df["Epoch"][:30], df["Tau"][:30], color=colors[idx], label=legend_labels[name], linestyle=':')
        ax[1].plot(df["Epoch"][30:], df["Tau"][30:], color=colors[idx], linestyle='--')

    elif name == "unst_cnn_cifar10_lth_real10":
        # 실선 (-) → 대쉬 (--)
        ax[1].plot(df["Epoch"][:30], df["Tau"][:30], color=colors[idx], label=legend_labels[name], linestyle='-')
        ax[1].plot(df["Epoch"][30:], df["Tau"][30:], color=colors[idx], linestyle='--')
    else:
        ax[1].plot(df["Epoch"], df["Tau"], color=colors[idx], label=legend_labels[name], linestyle='-')

# 축 및 제목 설정
ax[1].set_xlabel("Epoch", fontsize=9)
ax[1].set_ylabel(r"$\tau$", fontsize=9)
ax[1].grid(False)
ax[1].set_ylim(0, 1)
y_ticks = np.arange(0, 1.1, 0.2)
ax[1].set_yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks], fontsize=9)
ax[1].set_xlim(1, 200)
ax[1].tick_params(axis='x', labelsize=9)

ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

plt.savefig("cnn_all_sub.pdf", format="pdf", bbox_inches='tight')