import pandas as pd
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

desired_run_name_pattern = "s=7.0_v=1.2_tau=0.3"

data_list = []

for run in runs:
    if desired_run_name_pattern in run.name:
        data = {
            'run_id': run.id,
            'config': run.config,
            'history': run.history(keys=['test/epoch_acc']),
        }
        data_list.append(data)

runs_df = pd.DataFrame(data_list)

fig, ax = plt.subplots()

for index, row in runs_df.iterrows():
    x = range(len(row['history']['test/epoch_acc']))
    y = row['history']['test/epoch_acc']

mean_values = runs_df['history'].apply(lambda x: x['test/epoch_acc']).mean()
ci_lower = mean_values - 0.05
ci_upper = mean_values + 0.05
ax.plot(x, mean_values, label="Mean Test Accuracy", color='red', linestyle='--')

ax.fill_between(x, ci_lower, ci_upper, color='gray', alpha=0.2)

ax.set_xlabel('Epochs')
ax.set_ylabel('Test Accuracy')
ax.set_title('Shaded Line Plot of Test Accuracy with Mean Line')
ax.legend()
plt.show()
