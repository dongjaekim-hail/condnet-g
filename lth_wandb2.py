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
run_names = ["unst_mlp_mnist_lth_real10", "st_mlp_mnist_lth_real10"]
colors = ["purple", "green"]  # unst = purple, st = green

# 🔹 Run Name을 원하는 라벨로 변환
legend_labels = {
    "unst_mlp_mnist_lth_real10": "Unstructured LTH",
    "st_mlp_mnist_lth_real10": "Structured LTH"
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
    history = run.scan_history(keys=["test/epoch_acc"])

    epoch_acc_values = [row["test/epoch_acc"] for row in history if "test/epoch_acc" in row]

    # Epoch을 1부터 900까지 자동 생성
    epochs = list(range(1, len(epoch_acc_values) + 1))

    print(f"\n{name} - Total Data Points Retrieved: {len(epoch_acc_values)}")
    for epoch, value in zip(epochs[:10], epoch_acc_values[:10]):  # 처음 10개만 출력
        print(f"Epoch: {epoch}, Accuracy: {value}")

    dataframes[name] = pd.DataFrame({
        "Epoch": epochs,
        "Accuracy": epoch_acc_values
    })

# 7. 플롯 생성 (색상 및 커스텀 라벨 적용)
plt.figure(figsize=(10, 6))
for idx, (name, df) in enumerate(dataframes.items()):
    plt.plot(df["Epoch"], df["Accuracy"], color=colors[idx], label=legend_labels[name])

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy for Pruning Iterations (MLP LTH)")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.xlim(1, 900)  # Epoch 범위를 1~900으로 설정
plt.tight_layout()

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for name, df in dataframes.items():
    df.to_csv(f"{name}_test_epoch_acc_plot_data.csv", index=False)
    print(f"Data saved to {name}_test_epoch_acc_plot_data.csv")
