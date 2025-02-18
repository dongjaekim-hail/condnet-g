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
runs = api.runs("hails/condg_cnn")

# 4. 특정 run 찾기
run_names = [
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28",
    "st_cnn_cifar10_lth_real10"  # 추가된 Run
]
colors = ["blue", "black", "red", "orange", "green"]

legend_labels = {
    "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti": "CondGNet (Ours)",
    "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use": "CondNet",
    "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11": "Runtime Activation Magnitude",
    "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28": "Runtime Weight Magnitude",
    "st_cnn_cifar10_lth_real10": r"Structured Pruning (LTH) $\tau=0.4$"
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
for name, run in found_runs.items():
    if name == "st_cnn_cifar10_lth_real10":
        history = run.scan_history(keys=["Pruning Iteration 8/test/epoch_tau"])
        epoch_acc_values = [row["Pruning Iteration 8/test/epoch_tau"] for row in history if
                            "Pruning Iteration 8/test/epoch_tau" in row]
    else:
        history = run.scan_history(keys=["test/epoch_tau"])
        epoch_acc_values = [row["test/epoch_tau"] for row in history if "test/epoch_tau" in row]

    epochs = list(range(1, len(epoch_acc_values) + 1))

    if name == "st_cnn_cifar10_lth_real10":
        # 25~30 epoch의 mean 값 계산
        mean_tau = np.mean(epoch_acc_values[24:30])

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
plt.figure(figsize=(10, 6))
for idx, name in enumerate(run_names):
    df = dataframes[name]
    if name == "st_cnn_cifar10_lth_real10":
        plt.plot(df["Epoch"][:30], df["Tau"][:30], color=colors[idx], label=legend_labels[name], linestyle='-')  # 실선
        plt.plot(df["Epoch"][30:], df["Tau"][30:], color=colors[idx], linestyle='--')  # 점선
    else:
        plt.plot(df["Epoch"], df["Tau"], color=colors[idx], label=legend_labels[name], linestyle='-')


plt.xlabel("Epoch")
plt.ylabel("Test Tau")
plt.title("Test Tau for Multiple Runs (CNN)")
plt.legend(loc="lower right", fontsize=8, framealpha=0.8)
plt.grid(True)
plt.ylim(0, 1)
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.xlim(1, 200)
plt.tight_layout()

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장
for name, df in dataframes.items():
    df.to_csv(f"{name}_test_epoch_acc_plot_data.csv", index=False)
    print(f"Data saved to {name}_test_epoch_acc_plot_data.csv")
