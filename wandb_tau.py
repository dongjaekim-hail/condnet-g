import wandb
from wandb import Api

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

epoch_tau_values = history["test/epoch_tau"].dropna().tolist()
steps = history["_step"].dropna().tolist()

print("Step-wise 'test/epoch_tau' values:")
for step, value in zip(steps, epoch_tau_values):
    print(f"Step: {step}, test/epoch_tau: {value}")

import pandas as pd

df = pd.DataFrame({
    "Step": steps,
    "test/epoch_tau": epoch_tau_values
})

df.to_csv("test_epoch_tau_values.csv", index=False)
print("Data saved to test_epoch_tau_values.csv")
