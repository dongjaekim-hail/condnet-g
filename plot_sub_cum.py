import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# MLP와 CNN 데이터를 저장할 딕셔너리
cumulative_data_mlp = {}
cumulative_data_cnn = {}

scaling_factors = {
    "mlp": 535040,  # MLP Scaling Factor
    "cnn": 98306560  # CNN Scaling Factor
}

# MLP 및 CNN 관련 Run 이름 설정
run_names_mlp = [
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11",
    "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50",
    "unst_mlp_mnist_lth_real10",
    "st_mlp_mnist_lth_real10"
]

run_names_cnn = [
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28",
    "unst_cnn_cifar10_lth_real10",
    "st_cnn_cifar10_lth_real10"
]

# Found Runs 저장
found_runs_mlp = {}
found_runs_cnn = {}

# MLP Runs 가져오기
runs_mlp = api.runs("hails/condg_mlp")
for run in runs_mlp:
    if run.name in run_names_mlp:
        found_runs_mlp[run.name] = run
    if len(found_runs_mlp) == len(run_names_mlp):
        break

# CNN Runs 가져오기
runs_cnn = api.runs("hails/condg_cnn")
for run in runs_cnn:
    if run.name in run_names_cnn:
        found_runs_cnn[run.name] = run
    if len(found_runs_cnn) == len(run_names_cnn):
        break


# 데이터 처리 함수
def process_run_data(found_runs, run_names, scaling_factor):
    cumulative_data = {}
    pruning_iterations = 30

    for run_name in run_names[:4]:  # 일반적인 test/epoch_tau Runs
        history = found_runs[run_name].history(keys=["test/epoch_tau"])
        if "test/epoch_tau" in history:
            epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
            scaled_values = [value * scaling_factor for value in epoch_tau_values]
            cumulative_values = pd.Series(scaled_values).cumsum().tolist()
            total_steps = len(cumulative_values)
            normalized_steps = [s / total_steps for s in range(1, total_steps + 1)]
            cumulative_data[run_name] = (normalized_steps, cumulative_values)

    for run_name in [run_names[-2], run_names[-1]]:  # Pruning Iteration Runs
        cumulative_values = []
        for i in range(pruning_iterations):
            key = f"Pruning Iteration {i}/test/epoch_tau"
            history = found_runs[run_name].history(keys=[key])
            if key in history:
                epoch_tau_values = history[key].dropna().tolist()
                scaled_values = [value * scaling_factor for value in epoch_tau_values]
                if cumulative_values:
                    last_value = cumulative_values[-1]
                    cumulative_values.extend([last_value + val for val in pd.Series(scaled_values).cumsum()])
                else:
                    cumulative_values.extend(pd.Series(scaled_values).cumsum())
        total_steps = len(cumulative_values)
        normalized_steps = [s / total_steps for s in range(1, total_steps + 1)]
        cumulative_data[run_name] = (normalized_steps, cumulative_values)

    return cumulative_data


# MLP 및 CNN 데이터 처리
cumulative_data_mlp = process_run_data(found_runs_mlp, run_names_mlp, scaling_factors["mlp"])
cumulative_data_cnn = process_run_data(found_runs_cnn, run_names_cnn, scaling_factors["cnn"])

# 플롯 생성
fig, ax = plt.subplots(1, 2, figsize=(5.3, 2), width_ratios=[1, 1], constrained_layout=True)
colors = ["blue", "green", "red", "orange", "black", "black"]
linestyles = {"unst_mlp_mnist_lth_real10": "-", "st_mlp_mnist_lth_real10": ":", "unst_cnn_cifar10_lth_real10": "-",
              "st_cnn_cifar10_lth_real10": ":"}

legend_labels = {
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-": "CondGNet (Ours)",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use": "CondNet",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11": "Runtime Activation Magnitude",
    "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50": "Runtime Weight Magnitude",
    "unst_mlp_mnist_lth_real10": "Unstructured LTH",
    "st_mlp_mnist_lth_real10": "Structured LTH",
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti": "CondGNet (Ours)",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use": "CondNet",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11": "Runtime Activation Magnitude",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28": "Runtime Weight Magnitude",
    "unst_cnn_cifar10_lth_real10": "Unstructured LTH",
    "st_cnn_cifar10_lth_real10": "Structured LTH"
}

for i, (run_name, (x_vals, y_vals)) in enumerate(cumulative_data_mlp.items()):
    ax[0].plot(x_vals, y_vals, color=colors[i], linestyle=linestyles.get(run_name, "-"), label=legend_labels[run_name])

for i, (run_name, (x_vals, y_vals)) in enumerate(cumulative_data_cnn.items()):
    ax[1].plot(x_vals, y_vals, color=colors[i], linestyle=linestyles.get(run_name, "-"), label=legend_labels[run_name])

# 추가적인 플롯 설정
for i in range(2):
    ax[i].set_xlabel("Progress", fontsize=9)
    ax[i].grid(False)
    ax[i].set_ylim(bottom=0)
    ax[i].set_xlim(left=0, right=1)
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)

# X축 설정 (부동소수점 문제 해결)
for i in range(2):
    xticks = np.round(ax[i].get_xticks(), 2)
    xticks = [tick for tick in xticks if tick != 0.0]
    xtick_labels = [str(int(tick)) if np.isclose(tick, 1.0) else str(tick) for tick in xticks]
    ax[i].set_xticks(xticks)
    ax[i].set_xticklabels(xtick_labels, fontsize=9)

ax[0].set_ylabel(r"Cumulative GFlops", fontsize=9)
ax[1].set_ylabel(r"Cumulative GFlops", fontsize=9)

yticks = [0, 100000000, 200000000]
new_ylabels = ["0", "0.1", "0.2"]
ax[0].set_yticks(yticks)
ax[0].set_yticklabels(new_ylabels, fontsize=9)

# yticks, ylabels = ax[1].get_yticks(), ax[1].get_yticklabels()
# new_ylabels = [str(int(tick)) if tick == 0 else label.get_text() for tick, label in zip(yticks, ylabels)]
yticks = [0, 10000000000, 20000000000, 30000000000]
new_ylabels = ["0", "10", "20", "30"]
ax[1].set_yticks(yticks)
ax[1].set_yticklabels(new_ylabels, fontsize=9)

plt.savefig("cum_subs.pdf", format="pdf", bbox_inches='tight')
