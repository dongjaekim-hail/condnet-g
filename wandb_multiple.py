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
run_names = [
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11",
    "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50"
]
# run_names = [
#     "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti",
#     "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use",
#     "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11",
#     "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28"
# ]
colors = ["blue", "black", "red", "orange"]  # unst = purple, st = green

# 🔹 Run Name을 원하는 라벨로 변환
legend_labels = {
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-": "CondGNet (Ours)",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4_paper_use": "CondNet",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11": "Runtime Activation Magnitude",
    "mlp_runtime_weight_magnitude_tau=0.6_2025-02-18_15-57-50": "Runtime Weight Magnitude"
}
# legend_labels = {
#     "condg_cnn_schedule_s=7.0_v=0.5_tau=0.3_paper_ti": "CondGNet (Ours)",
#     "cond_cnn_schedule_s=7_v=0.5_tau=0.3_paper_use": "CondNet",
#     "cnn_runtime_activation_magnitude_tau=0.4_2024-12-08_15-08-11": "Runtime Activation Magnitude",
#     "cnn_runtime_weight_magnitude_tau=0.4_2024-12-08_15-08-28": "Runtime Weight Magnitude"
# }

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
    history = run.scan_history(keys=["test/epoch_tau"])

    epoch_acc_values = [row["test/epoch_tau"] for row in history if "test/epoch_tau" in row]

    # Epoch을 1부터 900까지 자동 생성
    epochs = list(range(1, len(epoch_acc_values) + 1))

    print(f"\n{name} - Total Data Points Retrieved: {len(epoch_acc_values)}")
    for epoch, value in zip(epochs[:10], epoch_acc_values[:10]):  # 처음 10개만 출력
        print(f"Epoch: {epoch}, Tau: {value}")

    dataframes[name] = pd.DataFrame({
        "Epoch": epochs,
        "Tau": epoch_acc_values
    })

# 7. 플롯 생성 (색상 및 커스텀 라벨 적용)
plt.figure(figsize=(10, 6))
for idx, name in enumerate(run_names):  # run_names 기준으로 순서 보장
    df = dataframes[name]
    plt.plot(df["Epoch"], df["Tau"], color=colors[idx], label=legend_labels[name])


plt.xlabel("Epoch")
plt.ylabel("Test Tau")
plt.title("Test Tau for Multiple Runs (MLP)")
plt.legend(loc="lower right", fontsize=8, framealpha=0.8)
plt.grid(True)
plt.ylim(0, 1)
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.xlim(1, 200)  # Epoch 범위를 1~900으로 설정
plt.tight_layout()

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for name, df in dataframes.items():
    df.to_csv(f"{name}_test_epoch_acc_plot_data.csv", index=False)
    print(f"Data saved to {name}_test_epoch_acc_plot_data.csv")
