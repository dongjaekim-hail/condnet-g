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
fig, ax = plt.subplots(figsize=(5.33, 3.198))  # fig, ax 추가
# cmap = plt.colormaps["Reds"]
# 최신 방식으로 컬러맵 및 정규화 설정
colors = ["blue", "yellow", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=30)  # 30단계

norm = mcolors.Normalize(vmin=0, vmax=29)

for i, (iteration, df) in enumerate(dataframes.items()):
    color = cmap(norm(i))
    ax.plot(df["Step"], df["Accuracy"], linestyle="-", label=iteration, color=color)

# # 컬러바 추가 (ax를 명확하게 지정)
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # 빈 배열 설정
# cbar = plt.colorbar(sm, ax=ax)  # ax 지정하여 오류 방지
# cbar.ax.tick_params(labelsize=9)  # 컬러바 눈금 폰트 크기 설정
# cbar.set_label("Pruning Iteration", fontsize=9)

# cbar_ax = fig.add_axes([0.82, 0.22, 0.02, 0.3])  # [left, bottom, width, height]
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # 빈 배열 설정
# cbar = plt.colorbar(sm, cax=cbar_ax)
# cbar.ax.tick_params(labelsize=9)  # 컬러바 눈금 폰트 크기 설정
# cbar.set_label("Pruning Iteration", fontsize=9)

# 축 및 제목 설정
ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("Accuracy", fontsize=9)
ax.tick_params(axis='both', labelsize=9)
ax.set_title("")
ax.grid(False)
ax.set_xlim(left=1, right=30)
ax.set_ylim(0, 1.0)  # y축 범위 설정
# ax.set_yticks([i / 10 for i in range(11)])  # y축 눈금 0.1 단위
y_ticks = np.arange(0, 1.1, 0.1)
ax.set_yticks(y_ticks)
ax.set_yticklabels(["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks])

plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 8. 플롯 출력
plt.show()

# 9. 결과를 CSV 파일로 저장 (선택 사항)
for iteration, df in dataframes.items():
    filename = f"{run_name}_{iteration.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")