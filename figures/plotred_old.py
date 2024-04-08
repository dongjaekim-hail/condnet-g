import pandas as pd
import wandb
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

target_tau = 0.6

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

cmap = sns.color_palette("Reds", as_cmap=True)

sns.set(style='whitegrid')

plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, cmap=cmap, cbar=True, cbar_kws={'label': 'test/epoch_acc'})
plt.xlabel('lambda_s')
plt.ylabel('lambda_v')
plt.title('test/epoch_acc Heatmap', fontsize=20)
plt.show()
