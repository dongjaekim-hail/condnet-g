import pandas as pd
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

desired_run_name_pattern = "s=7.0_v=1.2_tau=0.9"

data_list_acc = []
data_list_acc_bf = []

for run in runs:
    if desired_run_name_pattern in run.name:
        data_acc = {
            'run_id': run.id,
            'config': run.config,
            'history_acc': run.history(keys=['test/epoch_acc']),
            'history_acc_bf': run.history(keys=['test/epoch_acc_bf'])
        }
        data_list_acc.append(data_acc)
        data_list_acc_bf.append(data_acc)

runs_df_acc = pd.DataFrame(data_list_acc)
runs_df_acc_bf = pd.DataFrame(data_list_acc_bf)

fig, ax = plt.subplots()

# Calculate and plot the shaded area for 'test/epoch_acc'
mean_values_acc = runs_df_acc['history_acc'].apply(lambda x: x['test/epoch_acc']).mean()
ci_lower_acc = mean_values_acc - 0.05
ci_upper_acc = mean_values_acc + 0.05

x_acc = range(len(mean_values_acc))
ax.fill_between(x_acc, ci_lower_acc, ci_upper_acc, color='blue', alpha=0.1)

# Calculate and plot the shaded area for 'test/epoch_acc_bf'
mean_values_acc_bf = runs_df_acc_bf['history_acc_bf'].apply(lambda x: x['test/epoch_acc_bf']).mean()
ci_lower_acc_bf = mean_values_acc_bf - 0.05
ci_upper_acc_bf = mean_values_acc_bf + 0.05

x_acc_f = range(len(mean_values_acc_bf))
ax.fill_between(x_acc_f, ci_lower_acc_bf, ci_upper_acc_bf, color='red', alpha=0.1)

# Plot the mean lines
ax.plot(x_acc, mean_values_acc, label='epoch_acc', color='blue')
ax.plot(x_acc_f, mean_values_acc_bf, label='epoch_acc_bf', color='red')

ax.set_xlabel('Epochs')
ax.set_ylabel('Test Accuracy')
ax.set_title(f'epoch_acc_bf VS epoch_acc')
ax.legend()

plt.savefig(f'figures/bfaf0.9.png', dpi=300)