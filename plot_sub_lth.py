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



# 4. 특정 run 찾기
run_name_st = "st_mlp_mnist_lth_real10"
found_run = None

for run in runs:
    if run.name == run_name_st:
        found_run = run
        break

# 5. 해당 run이 없는 경우 에러 처리
if found_run is None:
    raise ValueError(f"Run '{run_name_st}'을 찾을 수 없습니다.")

# 6. Pruning Iteration 0~29까지의 test/epoch_acc 데이터 추출
dataframes_st = {}
pruning_iterations = [f"Pruning Iteration {i}/test/epoch_acc" for i in range(30)]

for iteration in pruning_iterations:
    history = found_run.history(keys=[iteration])
    if iteration in history.columns:
        acc_values = history[iteration].dropna().tolist()
        steps = list(range(1, len(acc_values) + 1))
        dataframes_st[iteration] = pd.DataFrame({
            "Step": steps,
            "Accuracy": acc_values
        })

# 7. 플롯 생성
fig, ax = plt.subplots(1,3, figsize=(5.3, 2), width_ratios=[1,1,1], constrained_layout=True)
fig.tight_layout(pad=1)
 # fig, ax 추가
# cmap = plt.colormaps["Reds"]
# 최신 방식으로 컬러맵 및 정규화 설정
# colors = ["blue", "yellow", "red"]
# cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=30)  # 30단계
cmap = cm.gray  # Grayscale 컬러맵

norm = mcolors.Normalize(vmin=0, vmax=29)

for i, (iteration, df) in enumerate(dataframes.items()):
    color = cmap(norm(i))
    ax[0].plot(df["Step"], df["Accuracy"], linestyle="-", label=iteration, color=color)

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
ax[0].set_xlabel("Epoch", fontsize=9)
ax[0].set_ylabel("Accuracy", fontsize=9)
ax[0].tick_params(axis='both', labelsize=9)
ax[0].grid(False)
ax[0].set_xlim(left=1, right=30)
ax[0].set_xticks([10, 20, 30], labels=["10", "20", "30"], fontsize=9)
ax[0].set_ylim(0, 1.0)  # y축 범위 설정
# ax.set_yticks([i / 10 for i in range(11)])  # y축 눈금 0.1 단위
y_ticks = np.arange(0, 1.1, 0.2)
ax[0].set_yticks(y_ticks)
ax[0].set_yticklabels(["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks])

# plt.tight_layout()
# plt.draw()
# ax = plt.gca()
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)


for i, (iteration, df) in enumerate(dataframes_st.items()):
    color = cmap(norm(i))
    ax[1].plot(df["Step"], df["Accuracy"], linestyle="-", label=iteration, color=color)

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
ax[1].set_xlabel("Epoch", fontsize=9)
# ax[1].set_ylabel("Accuracy", fontsize=9)
ax[1].tick_params(axis='both', labelsize=9)
ax[1].grid(False)
ax[1].set_xlim(left=1, right=30)
ax[1].set_xticks([10, 20, 30], labels=["10", "20", "30"], fontsize=9)
ax[1].set_ylim(0, 1.0)  # y축 범위 설정
# ax.set_yticks([i / 10 for i in range(11)])  # y축 눈금 0.1 단위
y_ticks = np.arange(0, 1.1, 0.2)
ax[1].set_yticks(y_ticks)
ax[1].set_yticklabels(["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks])

# plt.tight_layout()
# plt.draw()
# ax = plt.gca()
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

# 8. 플롯 출력
# plt.show()
# plt.savefig("cnn_unst_lth_acc.pdf", format="pdf", bbox_inches = 'tight')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 컬러맵 및 정규화 설정 (gray() 사용)
cmap = cm.gray  # Grayscale 컬러맵
norm = mcolors.Normalize(vmin=0, vmax=29)

fontsize = 9
# 컬러바 추가plt.colorbar(im, ax=axes.ravel().tolist())
cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax.ravel().tolist()[:2])  # ax 지정하여 오류 방지
cbar.ax.tick_params(labelsize=9)  # 컬러바 눈금 폰트 크기 설정
cbar.set_label("Pruning Iteration", fontsize=fontsize)
# 불필요한 축 제거
# ax.remove()
# fig.subplots_adjust(top=0.9, bottom=0.1)  # top과 bottom을 조정하여 y축 여백 추가



# ----

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
    "unst_mlp_mnist_lth_real10": "Unstructured",
    "st_mlp_mnist_lth_real10": "Structured"
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
# plt.figure(figsize=(1.5, 2))
linestyles = ["-", ":"]  # unst = 실선, st = 점선
for idx, name in enumerate(run_names):  # run_names 순서대로 데이터 사용
    df = dataframes[name]
    ax[2].plot(df["Epoch"], df["Tau"], color="black", linestyle=linestyles[idx], label=legend_labels[name])

ax[2].set_xlabel("Epoch", fontsize=9)
ax[2].set_ylabel(r"$\tau$", fontsize=9)
# ax[2].set_legend(fontsize=9, framealpha=0.8)
ax[2].grid(False)
ax[2].set_ylim(0, 1)
y_ticks = np.arange(0, 1.1, 0.2)
ax[2].set_yticks(y_ticks, labels=["0" if i == 0 else f"{i:.1f}" if i != 1 else "1" for i in y_ticks], fontsize=9)
ax[2].set_xlim(1, 900)  # Epoch 범위를 1~900으로 설정
ax[2].set_xticks([300, 600, 900], labels=["300", "600", "900"], fontsize=9)

ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)

plt.savefig("mlp_lth_subs.pdf", format="pdf", bbox_inches='tight')