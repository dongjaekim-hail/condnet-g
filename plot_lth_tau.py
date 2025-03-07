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
# colors = ["purple", "green"]  # unst = purple, st = green

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
plt.figure(figsize=(1.5, 2))
linestyles = ["-", ":"]  # unst = 실선, st = 점선
for idx, name in enumerate(run_names):  # run_names 순서대로 데이터 사용
    df = dataframes[name]
    plt.plot(df["Epoch"], df["Tau"], color="black", linestyle=linestyles[idx], label=legend_labels[name])

plt.xlabel("Epoch", fontsize=9)
plt.ylabel(r"$\tau$", fontsize=9)
plt.title("")
# plt.legend(fontsize=9, framealpha=0.8)
plt.grid(False)
plt.ylim(0, 1)
y_ticks = np.arange(0, 1.1, 0.2)
plt.yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks], fontsize=9)
plt.xlim(1, 900)  # Epoch 범위를 1~900으로 설정
plt.xticks([450, 900], labels=["450", "900"], fontsize=9)
plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 8. 플롯 출력
# plt.show()
plt.savefig("mlp_lth_tau.pdf", format="pdf", bbox_inches = 'tight')

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for name, df in dataframes.items():
    df.to_csv(f"{name}_test_epoch_acc_plot_data.csv", index=False)
    print(f"Data saved to {name}_test_epoch_acc_plot_data.csv")