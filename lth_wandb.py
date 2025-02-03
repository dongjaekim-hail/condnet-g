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

# 4. 특정 run 찾기
run_name = "unst_mlp_mnist_lth_real10"
found_run = None

for run in runs:
    if run.name == run_name:
        found_run = run
        break

# 5. 해당 run이 없는 경우 에러 처리
if found_run is None:
    raise ValueError(f"Run '{run_name}'을 찾을 수 없습니다.")

# 6. Pruning Iteration 0~29까지의 test/epoch_acc 데이터 추출
dataframes = {}
pruning_iterations = [f"Pruning Iteration {i}/test/epoch_acc" for i in range(30)]

for iteration in pruning_iterations:
    history = found_run.history(keys=[iteration])
    if iteration in history.columns:
        acc_values = history[iteration].dropna().tolist()
        steps = list(range(1, len(acc_values) + 1))
        dataframes[iteration] = pd.DataFrame({
            "Step": steps,
            "Accuracy": acc_values
        })

# 7. 플롯 생성
plt.figure(figsize=(12, 8))
for iteration, df in dataframes.items():
    plt.plot(df["Step"], df["Accuracy"], marker="o", linestyle="-", label=iteration)

plt.xlabel("Step")
plt.ylabel("Test Accuracy")
plt.title(f"Test Accuracy for Different Pruning Iterations ({run_name})")
plt.legend(fontsize=8, loc='best', ncol=2)  # 범례 조정
plt.grid(True)
plt.xlim(left=1)
plt.xlim(right=30)
plt.ylim(0, 1.0)  # y축 범위 설정
plt.yticks([i / 10 for i in range(11)])  # y축 눈금 0.1 단위
plt.tight_layout()

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for iteration, df in dataframes.items():
    filename = f"{run_name}_{iteration.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
