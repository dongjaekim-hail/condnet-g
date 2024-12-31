import wandb
from wandb import Api
import pandas as pd
import matplotlib.pyplot as plt

wandb.login()

api = Api()

runs = api.runs("hails/condg_mlp")

target_run = None
for run in runs:
    if run.name == "cond_mlp_schedule_s=7_v=0.2_tau=0.4":
        target_run = run
        break

if target_run is None:
    raise ValueError("해당 이름의 run을 찾을 수 없습니다.")

history = target_run.history(keys=["test/epoch_tau"])

# epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
# steps = history["_step"].dropna().tolist()
#
# # print("Step-wise 'test/epoch_tau' values:")
# # for step, value in zip(steps, epoch_tau_values):
# #     print(f"Step: {step}, test/epoch_tau: {value}")
#
# print("Step-wise 'test/epoch_tau' values with 0.7 multiplier:")
# for step, value in zip(steps, epoch_tau_values):
#     scaled_value = value * 0.7
#     print(f"Step: {step}, test/epoch_tau (scaled): {scaled_value}")
#
# import pandas as pd
#
# df = pd.DataFrame({
#     "Step": steps,
#     "test/epoch_tau": epoch_tau_values
# })
#
# # 파일 저장
# df.to_csv("test_epoch_tau_values.csv", index=False)
# print("Data saved to test_epoch_tau_values.csv")

epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
steps = history["_step"].dropna().tolist()
scaled_values = [value * 0.7 for value in epoch_tau_values]

plot_df = pd.DataFrame({
    "Step": steps,
    "Scaled Value": scaled_values
})

plt.figure(figsize=(10, 6))
plt.plot(plot_df["Step"], plot_df["Scaled Value"], marker="o", linestyle="-", label="Scaled test/epoch_tau")
plt.xlabel("Step")
plt.ylabel("Scaled Value")
plt.title("Scaled test/epoch_tau vs Step")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

plot_df.to_csv("test_epoch_tau_scaled_plot_data.csv", index=False)
print("Data saved to test_epoch_tau_scaled_plot_data.csv")
