import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)


x = np.arange(1,41)
for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern2 = "cond_3times_tau="+tau

    data_list2 = []
    for run in runs:
        if desired_run_name_pattern2 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_acc']),
            }
            data_list2.append(data)
    data_list2 = data_list2[-10:]
    runs_df2 = pd.DataFrame(data_list2)
print('')