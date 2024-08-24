import pandas as pd
import wandb
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import numpy as np

# make dirs
os.makedirs('../csvs/figures/', exist_ok=True)

api = wandb.Api()
entity, project = "hails", "condgnet_edit7"
runs = api.runs(entity + "/" + project)

# mins_acc = [0.86, 0.86, 0.94]
# maxs_acc = [0.96, 0.96, 0.96]

mins_acc = [0.8, 0.8, 0.8]
maxs_acc = [1.0, 1.0, 1.0]

target_step = 603  # Target step to filter

for count, target_tau in enumerate([0.6]):

    data_list = []

    for run in runs:
        res = json.loads(run.json_config)
        if res['tau']['value'] == target_tau:
            # Extract the test/epoch_acc data
            epoch_acc_data = run.history(keys=['test/epoch_acc', '_step'])

            # Filter for the specific step (603)
            acc_at_603 = epoch_acc_data[epoch_acc_data['_step'] == target_step]['test/epoch_acc'].values

            if len(acc_at_603) > 0:
                acc_at_603 = acc_at_603[0]
            else:
                acc_at_603 = np.nan  # If the step doesn't exist, set to NaN

            data = {
                'test/epoch_acc': acc_at_603,
                'lambda_s': run.config.get('lambda_s'),
                'lambda_v': run.config.get('lambda_v')
            }
            data_list.append(data)

    # Convert to DataFrame
    runs_df = pd.DataFrame(data_list)

    # Group by lambda_s and lambda_v, and select the row with the maximum test/epoch_acc for each group
    best_runs_df = runs_df.groupby(['lambda_s', 'lambda_v']).apply(
        lambda x: x.loc[x['test/epoch_acc'].idxmax()]).reset_index(drop=True)

    # Generate the heatmap using the best data
    heatmap_data = best_runs_df.pivot_table(index='lambda_v', columns='lambda_s', values='test/epoch_acc')

    heatmap_data = heatmap_data.reindex(heatmap_data.index[::-1])

    norm = matplotlib.colors.Normalize(0, 1)
    colors = [[norm(0), "white"],
              [norm(0.55), "pink"],
              [norm(0.65), "red"],
              [norm(1.0), "darkred"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    sns.set(style='whitegrid')

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.family'] = 'Helvetica'

    axs = sns.heatmap(heatmap_data, cmap=cmap, cbar=True, cbar_kws={'label': 'Accuracy'})
    axs.collections[0].set_clim(mins_acc[count], maxs_acc[count])
    ax.set_xlabel('$\lambda_s$')
    ax.set_ylabel('$\lambda_s$')

    idx1, idx2 = np.unravel_index(np.argmax(heatmap_data.to_numpy()), heatmap_data.to_numpy().shape)
    ax.scatter(idx2 + 0.5, idx1 + 0.5, marker='*', s=150, color='Yellow')

    plt.savefig(f'figures/heatmap_{target_tau}.png', dpi=300)