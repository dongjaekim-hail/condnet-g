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
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

mins_acc = [0.86, 0.86, 0.94]
maxs_acc = [0.96, 0.96, 0.96]

mins_acc = [0.9, 0.9, 0.9]
maxs_acc = [1, 1, 1]
for count, target_tau in enumerate([0.3,0.6, 0.9]):

    data_list = []

    for run in runs:
        res = json.loads(run.json_config)
        if res['tau']['value'] == target_tau:
            data = {
                'test/epoch_acc': run.summary.get('test/epoch_acc'),
                'lambda_s': run.config.get('lambda_s'),
                'lambda_v': run.config.get('lambda_v')
            }
            data_list.append(data)

    runs_df = pd.DataFrame(data_list)

    best_accuracy = runs_df['test/epoch_acc'].max()
    best_accuracy_index = runs_df['test/epoch_acc'].idxmax()

    heatmap_data = runs_df.pivot_table(index='lambda_v', columns='lambda_s', values='test/epoch_acc')

    heatmap_data = heatmap_data.reindex(heatmap_data.index[::-1])

    norm = matplotlib.colors.Normalize(0,1)
    colors = [[norm(0), "white"],
              # [norm(0.2), "grey"],
              [norm(0.55), "pink"],
              [norm(0.65), "red"],
              [norm(1.0), "darkred"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    sns.set(style='whitegrid')

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plt.rcParams.update({'font.size': 14})
    # set default font to helvetica
    plt.rcParams['font.family'] = 'Helvetica'

    axs = sns.heatmap(heatmap_data, cmap=cmap, cbar=True, cbar_kws={'label': 'Accuracy'})
    axs.collections[0].set_clim(mins_acc[count], maxs_acc[count])
    ax.set_xlabel('$\lambda_s$')
    ax.set_ylabel('$\lambda_s$')
    # plt.xlabel('$\lambda_s$')
    # plt.ylabel('$\lambda_v$')

    idx1,idx2 = np.unravel_index(np.argmax(heatmap_data.to_numpy()), heatmap_data.to_numpy().shape)
    # overlay star on best accuracy on the heatmap. which is 1,1
    ax.scatter(idx2+0.5, idx1+0.5, marker='*', s=150, color='Yellow')

    # find index (in x,y) where it maximum from heatmap_data.to_numpy()

    # save figure
    plt.savefig(f'figures/heatmap_{target_tau}.png', dpi=300)