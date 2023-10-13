import pandas as pd
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

desired_run_name_pattern1 = "s=7.0_v=1.2_tau=0.3"
desired_run_name_pattern2 = "cond_tau=0.3"

data_list1 = []
data_list2 = []

for run in runs:
    if desired_run_name_pattern1 in run.name:
        data = {
            'run_id': run.id,
            'config': run.config,
            'history': run.history(keys=['test/epoch_acc']),
        }
        data_list1.append(data)
    elif desired_run_name_pattern2 in run.name:
        data = {
            'run_id': run.id,
            'config': run.config,
            'history': run.history(keys=['test/epoch_acc']),
        }
        data_list2.append(data)

runs_df1 = pd.DataFrame(data_list1)
runs_df2 = pd.DataFrame(data_list2)

fig, ax = plt.subplots()

for index, row in runs_df1.iterrows():
    x = range(len(row['history']['test/epoch_acc']))
    y = row['history']['test/epoch_acc']

mean_values1 = runs_df1['history'].apply(lambda x: x['test/epoch_acc']).mean()
ci_lower1 = mean_values1 - 0.05
ci_upper1 = mean_values1 + 0.05
ax.plot(x, mean_values1, label="Condg Net tau 0.3", color='red')

ax.fill_between(x, ci_lower1, ci_upper1, color='red', alpha=0.1)

for index, row in runs_df2.iterrows():
    x = range(len(row['history']['test/epoch_acc']))
    y = row['history']['test/epoch_acc']

mean_values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).mean()
ci_lower2 = mean_values2 - 0.05
ci_upper2 = mean_values2 + 0.05
ax.plot(x, mean_values2, label="Cond Net tau 0.3", color='blue')

ax.fill_between(x, ci_lower2, ci_upper2, color='blue', alpha=0.1)

ax.set_xlabel('Epochs')
ax.set_ylabel('Test Accuracy')
ax.set_title('Condg Net VS Cond Net')
ax.legend()

plt.savefig(f'figures/shaded_0.3.png', dpi=300)
