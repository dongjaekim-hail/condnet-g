from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = '../csvs/condg-cond-stability/'

redish_cs = ['#FFC0CB', '#FF6347', '#B22222']
bluish_cs = ['#D3D3D3', '#A9A9A9', '#4B4B4B']
# plot figure using matplotlib using errorbar
fig = plt.figure(figsize=(6,5))
# set default fontsize
plt.rcParams.update({'font.size': 14})
# set default font to helvetica
plt.rcParams['font.family'] = 'Helvetica'
# matplotlib setting change to have no default legend
# ax.get_legend().remove()
plt.legend('', frameon=False)

for count, tau in enumerate([0.3, 0.6, 0.9]):
	# stability: 같은 데이터에 대해서 같은 뉴런이 얼마나 variance가 큰지 (클수록 안좋음)
	cond_= pd.read_csv(dir + 'condnet_stability_var_tau' + str(tau) + '.csv')

	mean_cond_ = np.array([cond_['layer1_mean'].values,cond_['layer2_mean'].values,cond_['layer3_mean'].values]).reshape(-1)
	std_cond_  = np.array([cond_['layer1_std'].values,cond_['layer2_std'].values,cond_['layer3_std'].values]).reshape(-1)
	plt.plot(np.arange(3), mean_cond_, color=bluish_cs[count], label=r"$\tau$={}".format(tau))
	plt.errorbar(np.arange(3), mean_cond_, fmt='.', yerr=std_cond_, color=bluish_cs[count], label='', capsize=5)

for count, tau in enumerate([0.3, 0.6, 0.9]):
	condg_ = pd.read_csv(dir + 'condgnet_stability_var_tau' + str(tau) + '.csv')

	mean_condg_ = np.array([condg_['layer1_mean'].values,condg_['layer2_mean'].values,condg_['layer3_mean'].values]).reshape(-1)
	std_condg_  = np.array([condg_['layer1_std'].values,condg_['layer2_std'].values,condg_['layer3_std'].values]).reshape(-1)
	plt.plot(np.arange(3), mean_condg_, color=redish_cs[count], label=r"$\tau$={}".format(tau))
	plt.errorbar(np.arange(3), mean_condg_, fmt='.', yerr=std_condg_, color=redish_cs[count], label='', capsize=5)
	print('')

plt.xlim(-0.25,2.25)
plt.xticks(np.arange(3), ['Layer 1', 'Layer 2', 'Layer 3'])

# plt.ylim(0,1.0)
# add legend in 2x3 grid but with small ticks and room
plt.legend(
    loc=(0.6, 0.12),         # Location
    ncols=2,
    handlelength=0.8,           # Line length
    handletextpad=0.5,        # Space between line and text
    borderaxespad=0,          # Space between legend and axes
    borderpad=0.5,            # Internal padding of legend
    fontsize='small',          # Font size
    frameon=False
)
# set yticks fro 0.05~ 0.25
plt.yticks(np.arange(0.05, 0.26, 0.05))
plt.ylim(0.075, 0.25)
# ax.legend(loc=(0.8, 0.4), ncol=2, fancybox=True, shadow=True)

# add CondGNet and CondNet in red text
ax = plt.gca()
plt.text(0.62, 0.33, 'CondNet', color='k', transform=ax.transAxes, fontsize='12')
plt.text(0.8, 0.33, 'CondGNet', color='r', transform=ax.transAxes, fontsize='12')
plt.ylabel('Variance of policy $\pi$ across simulations')
# save figure
plt.savefig(f'figures/stability_all.png', dpi=300)


# fig = plt.figure(figsize=(6,5))
# # set default fontsize
# plt.rcParams.update({'font.size': 14})
# # set default font to helvetica
# plt.rcParams['font.family'] = 'Helvetica'
# # matplotlib setting change to have no default legend
# # ax.get_legend().remove()
# plt.legend('', frameon=False)
#
# for count, tau in enumerate([0.3, 0.6, 0.9]):
# 	# adaptabiltiy: 뉴런간 variance가 얼마나 큰지 (클수록 다양하게 쓰는 것이므로 좋음)
# 	cond_adap_ = pd.read_csv(dir + 'condnet_adap_stability_var_tau' + str(tau) + '.csv')
#
# 	mean_cond_adap_ = np.array([cond_adap_['layer1_mean'].values,cond_adap_['layer2_mean'].values,cond_adap_['layer3_mean'].values]).reshape(-1)
# 	std_cond_adap_  = np.array([cond_adap_['layer1_std'].values,cond_adap_['layer2_std'].values,cond_adap_['layer3_std'].values]).reshape(-1)
# 	plt.plot(np.arange(3), mean_cond_adap_, color=bluish_cs[count], label=r"$\tau$={}".format(tau))
# 	plt.errorbar(np.arange(3), mean_cond_adap_, yerr=std_cond_adap_, color=bluish_cs[count], label='', capsize=5)
#
# for count, tau in enumerate([0.3, 0.6, 0.9]):
# 	condg_adap_ = pd.read_csv(dir + 'condgnet_adap_stability_var_tau' + str(tau) + '.csv')
#
# 	mean_condg_adap_ = np.array([condg_adap_['layer1_mean'].values,condg_adap_['layer2_mean'].values,condg_adap_['layer3_mean'].values]).reshape(-1)
# 	std_condg_adap_  = np.array([condg_adap_['layer1_std'].values,condg_adap_['layer2_std'].values,condg_adap_['layer3_std'].values]).reshape(-1)
# 	plt.plot(np.arange(3), mean_condg_adap_, color=redish_cs[count], label=r"$\tau$={}".format(tau))
# 	plt.errorbar(np.arange(3), mean_condg_adap_, yerr=std_condg_adap_, color=redish_cs[count], label='', capsize=5)
#
# 	print('')
#
# plt.xlim(-0.25,2.25)
# plt.xticks(np.arange(3), ['Layer 1', 'Layer 2', 'Layer 3'])
#
# # plt.ylim(0,1.0)
# # add legend in 2x3 grid but with small ticks and room
# plt.legend(
#     loc=(0.6, 0.12),         # Location
#     ncols=2,
#     handlelength=0.8,           # Line length
#     handletextpad=0.5,        # Space between line and text
#     borderaxespad=0,          # Space between legend and axes
#     borderpad=0.5,            # Internal padding of legend
#     fontsize='small',          # Font size
#     frameon=False
# )
#
# # ax.legend(loc=(0.8, 0.4), ncol=2, fancybox=True, shadow=True)
#
# # add CondGNet and CondNet in red text
# ax = plt.gca()
# plt.text(0.62, 0.33, 'CondNet', color='k', transform=ax.transAxes, fontsize='12')
# plt.text(0.8, 0.33, 'CondGNet', color='r', transform=ax.transAxes, fontsize='12')
#
#
#
# print('')
