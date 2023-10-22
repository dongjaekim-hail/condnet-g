import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)


num_batch_per_epoch = 500
num_epoch = 40

numparam_mlp = 535818
numparam_cond = 1592320
numparam_condg = 66305


# pick two colors with three different gradients
redish_cs = ['#FFC0CB', '#FF6347', '#B22222']
bluish_cs = ['#D3D3D3', '#A9A9A9', '#4B4B4B']
alpha = 0.14


layers = [784, 512, 256, 10]

def cal_num_weights(tau):
    return (784 * 512 + 512 * 256 + 256*10 )*tau


# backward
# set matplotlib settings (matplotlib.rcParams) to default
plt.rcdefaults()
fig = plt.figure(figsize=(6.7, 5))
# set default fontsize
plt.rcParams.update({'font.size': 14})
# set default font to helvetica
plt.rcParams['font.family'] = 'Helvetica'
# matplotlib setting change to have no default legend
# ax.get_legend().remove()
plt.legend('', frameon=False)

x = np.arange(1,41)
for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern2 = "cond_lastchance1024_tau="+tau

    data_list2 = []
    for run in runs:
        if desired_run_name_pattern2 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_tau']),
            }
            data_list2.append(data)
    data_list2 = data_list2[-10:]
    runs_df2 = pd.DataFrame(data_list2)

    x = np.arange(1,41)

    values2 = runs_df2['history'].apply(lambda x: x['test/epoch_tau']).to_numpy()
    values2 = cal_num_weights(values2) + numparam_cond
    values2 *= num_batch_per_epoch

    mean_values2 = values2.mean(axis=0)
    std_values   = values2.std(axis=0)
    ci_lower2 = mean_values2 - std_values
    ci_upper2 = mean_values2 + std_values
    plt.plot(x, mean_values2, label=r"$\tau$="+tau, color=bluish_cs[count])
    plt.fill_between(x, ci_lower2, ci_upper2, color=bluish_cs[count], alpha=alpha)

for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern1 = "s=7.0_v=1.2_tau="+tau
    data_list1 = []
    for run in runs:
        if desired_run_name_pattern1 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_tau']),
            }
            data_list1.append(data)
    data_list1 = data_list1[-10:]
    runs_df1 = pd.DataFrame(data_list1)

    x = np.arange(1,41)

    values1 = runs_df1['history'].apply(lambda x: x['test/epoch_tau']).to_numpy()
    values1 = cal_num_weights(values1) + numparam_condg
    values1 *= num_batch_per_epoch

    mean_values1 = values1.mean(axis=0)
    std_values   = values1.std(axis=0)
    ci_lower1 = mean_values1 - std_values
    ci_upper1 = mean_values1 + std_values
    plt.plot(x, mean_values1, label=r"$\tau$=" + tau, color=redish_cs[count])
    plt.fill_between(x, ci_lower1, ci_upper1, color=redish_cs[count], alpha=alpha)

    plt.xlabel('Epoch')
    plt.ylabel(r'$\tau$')

plt.xlabel('Epoch')
plt.ylabel('Number of parameters \nupdated in backpropagation')

plt.xlim(1,40)
# add legend in 2x3 grid but with small ticks and room
plt.legend(
    loc=(0.6, 0.3),         # Location
    ncols=2,
    handlelength=0.8,           # Line length
    handletextpad=0.5,        # Space between line and text
    borderaxespad=0,          # Space between legend and axes
    borderpad=0.5,            # Internal padding of legend
    fontsize='small',          # Font size
    frameon=False
)

# add CondGNet and CondNet in red text
ax = plt.gca()
plt.text(0.62, 0.51, 'CondNet', color='k', transform=ax.transAxes, fontsize='12')
plt.text(0.8, 0.51, 'CondGNet', color='r', transform=ax.transAxes, fontsize='12')

plt.savefig(f'figures/bwd_comput.png', dpi=300)


# set matplotlib settings (matplotlib.rcParams) to default
plt.rcdefaults()

fig = plt.figure(figsize=(6, 5))
# set default fontsize
plt.rcParams.update({'font.size': 14})
# set default font to helvetica
plt.rcParams['font.family'] = 'Helvetica'
# matplotlib setting change to have no default legend
# ax.get_legend().remove()
plt.legend('', frameon=False)
for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern2 = "cond_lastchance1024_tau="+tau

    data_list2 = []
    for run in runs:
        if desired_run_name_pattern2 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_tau']),
            }
            data_list2.append(data)
    data_list2 = data_list2[-10:]
    runs_df2 = pd.DataFrame(data_list2)

    x = np.arange(1,41)


    mean_values2 = runs_df2['history'].apply(lambda x: x['test/epoch_tau']).mean()
    std_values   = runs_df2['history'].apply(lambda x: x['test/epoch_tau']).std()
    ci_lower2 = mean_values2 - std_values
    ci_upper2 = mean_values2 + std_values
    plt.plot(x, mean_values2, label=r"$\tau$="+tau, color=bluish_cs[count])
    plt.fill_between(x, ci_lower2, ci_upper2, color=bluish_cs[count], alpha=alpha)

for count, tau in enumerate(['0.3', '0.6', '0.9']):
    desired_run_name_pattern1 = "s=7.0_v=1.2_tau="+tau
    data_list1 = []
    for run in runs:
        if desired_run_name_pattern1 in run.name:
            data = {
                'run_id': run.id,
                'config': run.config,
                'history': run.history(keys=['test/epoch_tau']),
            }
            data_list1.append(data)
    data_list1 = data_list1[-10:]
    runs_df1 = pd.DataFrame(data_list1)

    x = np.arange(1,41)

    mean_values1 = runs_df1['history'].apply(lambda x: x['test/epoch_tau']).mean()
    std_values = runs_df1['history'].apply(lambda x: x['test/epoch_tau']).std()
    ci_lower1 = mean_values1 - std_values
    ci_upper1 = mean_values1 + std_values
    plt.plot(x, mean_values1, label=r"$\tau$=" + tau, color=redish_cs[count])
    plt.fill_between(x, ci_lower1, ci_upper1, color=redish_cs[count], alpha=alpha)

    plt.xlabel('Epoch')
    plt.ylabel(r'$\tau$')

plt.xlim(1,40)
plt.ylim(0.3,1.0)
# add legend in 2x3 grid but with small ticks and room
plt.legend(
    loc=(0.6, 0.3),         # Location
    ncols=2,
    handlelength=0.8,           # Line length
    handletextpad=0.5,        # Space between line and text
    borderaxespad=0,          # Space between legend and axes
    borderpad=0.5,            # Internal padding of legend
    fontsize='small',          # Font size
    frameon=False
)

# ax.legend(loc=(0.8, 0.4), ncol=2, fancybox=True, shadow=True)

# add CondGNet and CondNet in red text
ax = plt.gca()
plt.text(0.62, 0.51, 'CondNet', color='k', transform=ax.transAxes, fontsize='12')
plt.text(0.8, 0.51, 'CondGNet', color='r', transform=ax.transAxes, fontsize='12')

plt.savefig(f'figures/shaded_tau_all.png', dpi=300)

# Number of parameters: 535818
# CONDNet Number of parameters: 1592320
# Number of parameters: 535818
# CONDG net : Number of parameters: 66305




# efficiency of the model
# set matplotlib settings (matplotlib.rcParams) to default
plt.rcdefaults()
fig = plt.figure(figsize=(6.7, 5))
# set default fontsize
plt.rcParams.update({'font.size': 14})
# set default font to helvetica
plt.rcParams['font.family'] = 'Helvetica'
# matplotlib setting change to have no default legend
# ax.get_legend().remove()
plt.legend('', frameon=False)

x = np.arange(1,41)

condg_fwd_params = np.arange(1,41) * num_batch_per_epoch *(numparam_mlp+numparam_mlp+numparam_condg)
cond_fwd_params = np.arange(1,41) * num_batch_per_epoch *(numparam_mlp+numparam_cond)


plt.plot(x, cond_fwd_params, label="CondNet", color='k')
plt.plot(x, condg_fwd_params, label="CondGNet", color='r')

plt.xlabel('Epoch')
plt.ylabel('Cumulative number of parameters \nused in feed-forward computation')

print(cond_fwd_params[-1] - condg_fwd_params[-1])

plt.xlim(1,40)
# add legend in 2x3 grid but with small ticks and room
plt.legend(
    loc=(0.7, 0.2),         # Location
    handlelength=0.8,           # Line length
    handletextpad=0.5,        # Space between line and text
    borderaxespad=0,          # Space between legend and axes
    borderpad=0.5,            # Internal padding of legend
    fontsize='small',          # Font size
    frameon=False
)


# add CondGNet and CondNet in red text
ax = plt.gca()
plt.savefig(f'figures/forward_comput.png', dpi=300)


