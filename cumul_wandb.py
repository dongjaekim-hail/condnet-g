# import wandb
# from wandb import Api
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 1. WandB 로그인
# wandb.login()
#
# # 2. W&B API 객체 생성
# api = Api()
#
# # 3. 원하는 run 찾기
# run_name = "unst_mlp_mnist_lth_real10"
# found_run = None
#
# runs = api.runs("hails/condg_mlp")
#
# for run in runs:
#     if run.name == run_name:
#         found_run = run
#         break
#
# # 4. 해당 run이 없을 경우 에러 처리
# if not found_run:
#     raise ValueError(f"Run {run_name}을 찾을 수 없습니다.")
#
# # 5. 로그 데이터 추출
# history = found_run.history(keys=["Pruning Iteration 1/test/epoch_tau"])
# epoch_tau_values = history["Pruning Iteration 1/test/epoch_tau"].dropna().tolist()
#
# # 6. Step 설정 및 X축 정규화
# steps = list(range(1, len(epoch_tau_values) + 1))  # [1, 2, ..., 30]
# normalized_steps = [(s - 1) / (len(steps) - 1) for s in steps]  # [0, 1/29, 2/29, ..., 1]
#
# # 7. 값 변환 및 누적 합 계산
# scaled_values = [value * 535040 for value in epoch_tau_values]
# cumulative_values = pd.Series(scaled_values).cumsum().tolist()
#
# # 8. 누적 그래프 생성 (X축 0~1)
# plt.figure(figsize=(10, 6))
# plt.plot(normalized_steps, cumulative_values, marker="o", linestyle="-", label="Cumulative Flops")
# plt.xlabel("Normalized Step (0 to 1)")
# plt.ylabel("Cumulative Flops")
# plt.title("Cumulative Flops over Normalized Steps")
# plt.legend()
# plt.grid(True)
# plt.xlim(0, 1)  # X축을 0~1로 설정
# plt.tight_layout()
#
# # 9. 그래프 출력
# plt.show()
#
# # 10. 결과를 CSV 파일로 저장
# df = pd.DataFrame({
#     "Normalized Step": normalized_steps,
#     "Cumulative Flops": cumulative_values
# })
# df.to_csv(f"{run_name}_normalized_cumulative_flops.csv", index=False)
# print(f"Data saved to {run_name}_normalized_cumulative_flops.csv")

# import wandb
# from wandb import Api
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 1. WandB 로그인
# wandb.login()
#
# # 2. W&B API 객체 생성
# api = Api()
#
# # 3. 원하는 run 찾기
# run_name = "unst_mlp_mnist_lth_real10"
# found_run = None
#
# runs = api.runs("hails/condg_mlp")
#
# for run in runs:
#     if run.name == run_name:
#         found_run = run
#         break
#
# # 4. 해당 run이 없을 경우 에러 처리
# if not found_run:
#     raise ValueError(f"Run {run_name}을 찾을 수 없습니다.")
#
# # 5. 전체 누적 값을 저장할 리스트
# cumulative_values = []
# scaling_factor = 535040  # Flops 스케일 조정
# pruning_iterations = 30  # Pruning Iteration 0 ~ 29
#
# # 6. Pruning Iteration 0 ~ 29까지 반복
# for i in range(pruning_iterations):
#     key = f"Pruning Iteration {i}/test/epoch_tau"  # 해당 iteration의 key
#     history = found_run.history(keys=[key])
#
#     # 데이터가 존재하면 가져오기
#     if key in history:
#         epoch_tau_values = history[key].dropna().tolist()
#         scaled_values = [value * scaling_factor for value in epoch_tau_values]
#
#         # **이전 누적 값에서 계속 누적하기**
#         if cumulative_values:
#             last_value = cumulative_values[-1]  # 마지막 누적 값
#             cumulative_values.extend([last_value + val for val in pd.Series(scaled_values).cumsum()])
#         else:
#             cumulative_values.extend(pd.Series(scaled_values).cumsum())
#
# # 7. Step 설정 (전체 데이터 개수를 0~1 사이로 변환)
# total_steps = len(cumulative_values)
# normalized_steps = [s / total_steps for s in range(1, total_steps + 1)]
#
# # 8. 누적 그래프 생성
# plt.figure(figsize=(10, 6))
# plt.plot(normalized_steps, cumulative_values, marker="o", linestyle="-", label="Cumulative Flops")
# plt.xlabel("Normalized Step (0 to 1)")
# plt.ylabel("Cumulative Flops")
# plt.title("Cumulative Flops over Normalized Steps with Pruning Iterations")
# plt.legend()
# plt.grid(True)
# plt.xlim(left=0, right=1)  # X축을 0~1 범위로 설정
# plt.tight_layout()
#
# # 9. 그래프 출력
# plt.show()
#
# # 10. 결과를 CSV 파일로 저장
# df = pd.DataFrame({
#     "Normalized Step": normalized_steps,
#     "Cumulative Flops": cumulative_values
# })
# df.to_csv(f"{run_name}_cumulative_flops_with_pruning.csv", index=False)
# print(f"Data saved to {run_name}_cumulative_flops_with_pruning.csv")

# import wandb
# from wandb import Api
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 1. WandB 로그인
# wandb.login()
#
# # 2. W&B API 객체 생성
# api = Api()
#
# # 3. 원하는 Runs 찾기
# run_names = [
#     "unst_mlp_mnist_lth_real10",
#     "cond_mlp_schedule_s=7_v=0.2_tau=0.4",
#     "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_80"
# ]
# found_runs = {}
#
# runs = api.runs("hails/condg_mlp")
#
# for run in runs:
#     if run.name in run_names:
#         found_runs[run.name] = run
#     if len(found_runs) == len(run_names):
#         break
#
# # 4. Run이 모두 존재하는지 확인
# if len(found_runs) != len(run_names):
#     raise ValueError("일부 지정된 run을 찾을 수 없습니다.")
#
# # 5. 누적 데이터를 저장할 딕셔너리
# cumulative_data = {}
#
# # 6. 첫 번째 Run (unst_mlp_mnist_lth_real10): Pruning Iteration 0~29 누적
# scaling_factor = 535040
# pruning_iterations = 30  # Pruning Iteration 0 ~ 29
# run_name_1 = "unst_mlp_mnist_lth_real10"
# cumulative_values = []
#
# for i in range(pruning_iterations):
#     key = f"Pruning Iteration {i}/test/epoch_tau"
#     history = found_runs[run_name_1].history(keys=[key])
#
#     if key in history:
#         epoch_tau_values = history[key].dropna().tolist()
#         scaled_values = [value * scaling_factor for value in epoch_tau_values]
#
#         if cumulative_values:
#             last_value = cumulative_values[-1]
#             cumulative_values.extend([last_value + val for val in pd.Series(scaled_values).cumsum()])
#         else:
#             cumulative_values.extend(pd.Series(scaled_values).cumsum())
#
# # X축 정규화
# total_steps_1 = len(cumulative_values)
# normalized_steps_1 = [s / total_steps_1 for s in range(1, total_steps_1 + 1)]
# cumulative_data[run_name_1] = (normalized_steps_1, cumulative_values)
#
# # 7. 두 번째 Run (cond_mlp_schedule_s=7_v=0.2_tau=0.4): test/epoch_tau 단순 누적
# run_name_2 = "cond_mlp_schedule_s=7_v=0.2_tau=0.4"
# history = found_runs[run_name_2].history(keys=["test/epoch_tau"])
#
# if "test/epoch_tau" in history:
#     epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
#     scaled_values = [value * scaling_factor for value in epoch_tau_values]
#     cumulative_values_2 = pd.Series(scaled_values).cumsum().tolist()
#
#     # X축 정규화
#     total_steps_2 = len(cumulative_values_2)
#     normalized_steps_2 = [s / total_steps_2 for s in range(1, total_steps_2 + 1)]
#     cumulative_data[run_name_2] = (normalized_steps_2, cumulative_values_2)
#
# # 8. 세 번째 Run (condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-): test/epoch_tau 단순 누적
# run_name_3 = "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_80"
# history = found_runs[run_name_3].history(keys=["test/epoch_tau"])
#
# if "test/epoch_tau" in history:
#     epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
#     scaled_values = [value * scaling_factor for value in epoch_tau_values]
#     cumulative_values_3 = pd.Series(scaled_values).cumsum().tolist()
#
#     # X축 정규화
#     total_steps_3 = len(cumulative_values_3)
#     normalized_steps_3 = [s / total_steps_3 for s in range(1, total_steps_3 + 1)]
#     cumulative_data[run_name_3] = (normalized_steps_3, cumulative_values_3)
#
# # 9. 세 개의 누적 그래프 출력
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_data[run_name_1][0], cumulative_data[run_name_1][1], marker="o", linestyle="-", label=run_name_1)
# plt.plot(cumulative_data[run_name_2][0], cumulative_data[run_name_2][1], marker="s", linestyle="--", label=run_name_2)
# plt.plot(cumulative_data[run_name_3][0], cumulative_data[run_name_3][1], marker="^", linestyle=":", label=run_name_3)
#
# plt.xlabel("Normalized Step (0 to 1)")
# plt.ylabel("Cumulative Flops")
# plt.title("Cumulative Flops over Normalized Steps for Three Runs")
# plt.legend()
# plt.grid(True)
# plt.xlim(left=0, right=1)  # X축을 0~1 범위로 설정
# plt.tight_layout()
#
# # 10. 그래프 출력
# plt.show()
#
# # 11. CSV 파일 저장
# df_1 = pd.DataFrame({
#     "Normalized Step": cumulative_data[run_name_1][0],
#     "Cumulative Flops": cumulative_data[run_name_1][1]
# })
# df_1.to_csv(f"{run_name_1}_cumulative_flops.csv", index=False)
#
# df_2 = pd.DataFrame({
#     "Normalized Step": cumulative_data[run_name_2][0],
#     "Cumulative Flops": cumulative_data[run_name_2][1]
# })
# df_2.to_csv(f"{run_name_2}_cumulative_flops.csv", index=False)
#
# df_3 = pd.DataFrame({
#     "Normalized Step": cumulative_data[run_name_3][0],
#     "Cumulative Flops": cumulative_data[run_name_3][1]
# })
# df_3.to_csv(f"{run_name_3}_cumulative_flops.csv", index=False)
#
# print(f"Data saved to {run_name_1}_cumulative_flops.csv")
# print(f"Data saved to {run_name_2}_cumulative_flops.csv")
# print(f"Data saved to {run_name_3}_cumulative_flops.csv")

import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt

# 1. WandB 로그인
wandb.login()

# 2. W&B API 객체 생성
api = Api()

# 3. 원하는 Runs 찾기
run_names = [
    "unst_mlp_mnist_lth_real10",
    "cond_mlp_schedule_s=7_v=0.2_tau=0.4",
    "condg_mlp_schedule_s=7.0_v=0.2_tau=0.3_paper_ti-",
    "mlp_runtime_activation_magnitude_tau=0.6_2024-12-09_17-23-11",
    "mlp_runtime_weight_magnitude_tau=0.6_2024-12-09_17-23-02"
]
found_runs = {}

runs = api.runs("hails/condg_mlp")

for run in runs:
    if run.name in run_names:
        found_runs[run.name] = run
    if len(found_runs) == len(run_names):
        break

# 4. Run이 모두 존재하는지 확인
if len(found_runs) != len(run_names):
    raise ValueError("일부 지정된 run을 찾을 수 없습니다.")

# 5. 누적 데이터를 저장할 딕셔너리
cumulative_data = {}
scaling_factor = 535040

# 6. 첫 번째 Run (unst_mlp_mnist_lth_real10): Pruning Iteration 0~29 누적
pruning_iterations = 30
run_name_1 = "unst_mlp_mnist_lth_real10"
cumulative_values = []

for i in range(pruning_iterations):
    key = f"Pruning Iteration {i}/test/epoch_tau"
    history = found_runs[run_name_1].history(keys=[key])

    if key in history:
        epoch_tau_values = history[key].dropna().tolist()
        scaled_values = [value * scaling_factor for value in epoch_tau_values]

        if cumulative_values:
            last_value = cumulative_values[-1]
            cumulative_values.extend([last_value + val for val in pd.Series(scaled_values).cumsum()])
        else:
            cumulative_values.extend(pd.Series(scaled_values).cumsum())

# X축 정규화
total_steps_1 = len(cumulative_values)
normalized_steps_1 = [s / total_steps_1 for s in range(1, total_steps_1 + 1)]
cumulative_data[run_name_1] = (normalized_steps_1, cumulative_values)

# 7. 나머지 Runs: test/epoch_tau 단순 누적
for run_name in run_names[1:]:  # 첫 번째 Run은 이미 처리했으므로 나머지를 처리
    history = found_runs[run_name].history(keys=["test/epoch_tau"])

    if "test/epoch_tau" in history:
        epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
        scaled_values = [value * scaling_factor for value in epoch_tau_values]
        cumulative_values = pd.Series(scaled_values).cumsum().tolist()

        # X축 정규화
        total_steps = len(cumulative_values)
        normalized_steps = [s / total_steps for s in range(1, total_steps + 1)]
        cumulative_data[run_name] = (normalized_steps, cumulative_values)

# # 8. 다섯 개의 누적 그래프 출력
# plt.figure(figsize=(10, 6))
# markers = ["o", "s", "^", "D", "x"]  # 각 라인의 마커 스타일 지정
# linestyles = ["-", "--", ":", "-.", "-"]  # 각 라인의 스타일 지정
#
# for i, (run_name, (x_vals, y_vals)) in enumerate(cumulative_data.items()):
#     plt.plot(x_vals, y_vals, marker=markers[i], linestyle=linestyles[i], label=run_name)

# 8. 다섯 개의 누적 그래프 출력 (색만 다르게, 선은 얇게)
plt.figure(figsize=(10, 6))
colors = ["blue", "red", "orange", "purple", "green"]  # 각 라인의 색 지정

for i, (run_name, (x_vals, y_vals)) in enumerate(cumulative_data.items()):
    plt.plot(x_vals, y_vals, color=colors[i], linewidth=0.5, label=run_name)  # 얇은 선, 색만 변경


plt.xlabel("Normalized Step (0 to 1)")
plt.ylabel("Cumulative Flops")
plt.title("Cumulative Flops over Normalized Steps for Multiple Runs")
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=1)  # X축을 0~1 범위로 설정
plt.tight_layout()

# 9. 그래프 출력
plt.show()

# 10. CSV 파일 저장
for run_name, (x_vals, y_vals) in cumulative_data.items():
    df = pd.DataFrame({
        "Normalized Step": x_vals,
        "Cumulative Flops": y_vals
    })
    df.to_csv(f"{run_name}_cumulative_flops.csv", index=False)
    print(f"Data saved to {run_name}_cumulative_flops.csv")
