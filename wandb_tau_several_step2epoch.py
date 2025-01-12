import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# 3. 원하는 프로젝트와 엔터티의 실행 목록 가져오기
runs = api.runs("hails/condg_mlp")

# 4. 특정 run 두 개 찾기
run_names = ["unst_mlp_mnist_lth", "cond_mlp_schedule_s=7_v=0.2_tau=0.4", "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11", "mlp_runtime_weight_magnitude_tau=0.6_2024-12-09_17-23-02", "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-"]
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
    if name == "unst_mlp_mnist_lth":
        history = run.history(keys=["Pruning Iteration 1/test/epoch_tau"])
        epoch_tau_values = history["Pruning Iteration 1/test/epoch_tau"].dropna().tolist()
    else:
        history = run.history(keys=["test/epoch_tau"])
        epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
    # steps = history["_step"].dropna().tolist()
    # Step을 1, 2, 3,... 순으로 설정
    steps = list(range(1, len(epoch_tau_values) + 1))
    scaled_values = [value * 535040 for value in epoch_tau_values]
    print(f"\n{name} Step-wise calculated values:")
    for step, value in zip(steps, scaled_values):
        print(f"Step: {step}, Scaled Value: {value}")
    dataframes[name] = pd.DataFrame({
        "Step": steps,
        "Scaled Value": scaled_values
    })

# 7. 플롯 생성
plt.figure(figsize=(10, 6))
for name, df in dataframes.items():
    plt.plot(df["Step"], df["Scaled Value"], marker="o", linestyle="-", label=name)

plt.xlabel("Step")
plt.ylabel("Flops")
plt.title("Flops for Multiple Runs")
plt.legend()
plt.grid(True)
plt.xlim(left=0)
plt.xlim(right=200)
plt.tight_layout()

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for name, df in dataframes.items():
    df.to_csv(f"{name}_test_epoch_tau_scaled_plot_data.csv", index=False)
    print(f"Data saved to {name}_test_epoch_tau_scaled_plot_data.csv")