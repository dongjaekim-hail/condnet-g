import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)


# empty data frame for taus and mean and std
data_ = pd.DataFrame(columns=['tau', 'mean', 'std'])

for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern2 = "cond_tau="+tau

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


    values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).to_numpy()
    mean_values2 = values2.mean(axis=0)
    std_valeus2 = values2.std(axis=0)

    data_.loc[count] = [tau, mean_values2[-1], std_valeus2[-1]]

data_condnet_withoutput = data_.copy()

data_ = pd.DataFrame(columns=['tau', 'mean', 'std'])

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


    values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).to_numpy()
    mean_values2 = values2.mean(axis=0)
    std_valeus2 = values2.std(axis=0)

    data_.loc[count] = [tau, mean_values2[-1], std_valeus2[-1]]

data_condnet_3times = data_.copy()

data_ = pd.DataFrame(columns=['tau', 'mean', 'std'])

for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern2 = "cond_lastchance1024_tau="+tau

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

    values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).to_numpy()
    mean_values2 = values2.mean(axis=0)
    std_valeus2 = values2.std(axis=0)

    data_.loc[count] = [tau, mean_values2[-1], std_valeus2[-1]]

data_condnet_without_output = data_.copy()

data_ = pd.DataFrame(columns=['tau', 'mean', 'std'])

for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern1 = "s=7.0_v=1.2_tau="+tau
    data_list2 = []
    for run in runs:
        if desired_run_name_pattern1 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_acc']),
            }
            data_list2.append(data)
    data_list2 = data_list2[-10:]
    runs_df2 = pd.DataFrame(data_list2)

    values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).to_numpy()
    mean_values2 = values2.mean(axis=0)
    std_valeus2 = values2.std(axis=0)

    data_.loc[count] = [tau, mean_values2[-1], std_valeus2[-1]]

data_condgnet_withoutput = data_.copy()

data_ = pd.DataFrame(columns=['tau', 'mean', 'std'])

for count, tau in enumerate(['0.3', '0.6', '0.9']):
    if tau == '0.9':
        desired_run_name_pattern1 = "s=7.0_v=1.2_tau="+tau
    else:
        desired_run_name_pattern1 = "condg_biggert_s=7.0_v=1.2_tau="+tau
    data_list2 = []
    for run in runs:
        if desired_run_name_pattern1 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_acc']),
            }
            data_list2.append(data)
    if tau == '0.9':
        data_list2 = data_list2[-20:-10]
    else:
        data_list2 = data_list2[-10:]

    runs_df2 = pd.DataFrame(data_list2)

    values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).to_numpy()
    mean_values2 = values2.mean(axis=0)
    std_valeus2 = values2.std(axis=0)

    data_.loc[count] = [tau, mean_values2[-1], std_valeus2[-1]]

data_condgnet_withoutput_bigger = data_.copy()

# primt mean and std for data_condnet_withoutput and data_condgnet_without_output
print('data_condnet_withoutput')
print(data_condnet_withoutput)

print('data_condnet_without_output')
print(data_condnet_without_output)

print('data_condgnet_withoutput')
print(data_condgnet_withoutput)

print('data_condnet_3times')
print(data_condnet_3times)

print('data_condgnet_withoutput_bigger')
print(data_condgnet_withoutput_bigger)