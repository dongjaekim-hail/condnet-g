import pandas as pd
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
entity, project = "hails", "condgnet"
runs = api.runs(entity + "/" + project)

# pick two colors with three different gradients
redish_cs = ['#FFA07A', '#FF0000', '#8B0000']
redish_cs = ['#FFC0CB', '#FF6347', '#B22222']
bluish_cs = ['#D3D3D3', '#A9A9A9', '#4B4B4B']
alpha = 0.14

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

    for index, row in runs_df2.iterrows():
        x = range(1, len(row['history']['test/epoch_acc'])+1)
        y = row['history']['test/epoch_acc']

    mean_values2 = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).mean()
    std_values   = runs_df2['history'].apply(lambda x: x['test/epoch_acc']).std()
    ci_lower2 = mean_values2 - std_values
    ci_upper2 = mean_values2 + std_values
    plt.plot(x, mean_values2, label=r"$\tau$="+tau, color=bluish_cs[count])
    plt.fill_between(x, ci_lower2, ci_upper2, color=bluish_cs[count], alpha=alpha)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


for count, tau in enumerate(['0.3', '0.6', '0.9']):

    desired_run_name_pattern1 = "s=7.0_v=1.2_tau="+tau

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

    # pick only last 10 runs
    data_list1 = data_list1[-10:]
    runs_df1 = pd.DataFrame(data_list1)

    for index, row in runs_df1.iterrows():
        x = range(1, len(row['history']['test/epoch_acc'])+1)
        y = row['history']['test/epoch_acc']

    mean_values1 = runs_df1['history'].apply(lambda x: x['test/epoch_acc']).mean()
    std_values1 = runs_df1['history'].apply(lambda x: x['test/epoch_acc']).std()
    ci_lower1 = mean_values1 - std_values1
    ci_upper1 = mean_values1 + std_values1

    plt.plot(x, mean_values1, label=r"$\tau$="+tau, color=redish_cs[count])
    plt.fill_between(x, ci_lower1, ci_upper1, color=redish_cs[count], alpha=alpha)

plt.xlim(1,40)
plt.ylim(0,1.0)
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

plt.savefig(f'figures/shaded_all.png', dpi=300)
