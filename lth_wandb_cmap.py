import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# 3. 원하는 프로젝트와 엔터티의 실행 목록 가져오기
runs = api.runs("hails/condg_mlp")

# 4. 특정 run 찾기
run_name = "st_mlp_mnist_lth_real10"
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
fig, ax = plt.subplots(figsize=(12, 8))  # fig, ax 추가
# cmap = plt.colormaps["Reds"]
# 최신 방식으로 컬러맵 및 정규화 설정
colors = ["blue", "yellow", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=30)  # 30단계

norm = mcolors.Normalize(vmin=0, vmax=29)

for i, (iteration, df) in enumerate(dataframes.items()):
    color = cmap(norm(i))
    ax.plot(df["Step"], df["Accuracy"], marker="o", linestyle="-", label=iteration, color=color)

# 컬러바 추가 (ax를 명확하게 지정)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 빈 배열 설정
cbar = plt.colorbar(sm, ax=ax)  # ax 지정하여 오류 방지
cbar.set_label("Pruning Iteration")

# 축 및 제목 설정
ax.set_xlabel("Step")
ax.set_ylabel("Test Accuracy")
ax.set_title(f"Test Accuracy for Different Pruning Iterations ({run_name})")
ax.grid(True)
ax.set_xlim(left=1, right=30)
ax.set_ylim(0, 1.0)  # y축 범위 설정
ax.set_yticks([i / 10 for i in range(11)])  # y축 눈금 0.1 단위
plt.tight_layout()

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for iteration, df in dataframes.items():
    filename = f"{run_name}_{iteration.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
