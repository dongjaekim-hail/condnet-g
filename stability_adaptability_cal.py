from glob import glob
import numpy as np
import pandas as pd

dir = 'condg-cond-stability/'

for tau in [0.3, 0.6, 0.9]:
	# empty dataframe for each layer
	cond_ = pd.DataFrame(columns=['layer1_mean', 'layer1_std', 'layer2_mean', 'layer2_std', 'layer3_mean', 'layer3_std'])
	condg_ = pd.DataFrame(columns=['layer1_mean', 'layer1_std', 'layer2_mean', 'layer2_std', 'layer3_mean', 'layer3_std'])

	cond_adap_ = pd.DataFrame(columns=['layer1_mean', 'layer1_std', 'layer2_mean', 'layer2_std', 'layer3_mean', 'layer3_std'])
	condg_adap_ = pd.DataFrame(columns=['layer1_mean', 'layer1_std', 'layer2_mean', 'layer2_std', 'layer3_mean', 'layer3_std'])

	cond_VAR_mean = []
	cond_VAR_std = []
	condg_VAR_mean = []
	condg_VAR_std = []

	cond_adap_VAR_mean = []
	cond_adap_VAR_std = []
	condg_adap_VAR_mean = []
	condg_adap_VAR_std = []

	for layer in [0,1,2]:
		cond_layer = []
		condg_layer = []

		cond_name = dir + 'condnet_iter*tau' + str(tau) + '*layer{}.npy'.format(layer)
		condg_name = dir + 'condgnet_iter*tau' + str(tau) + '*layer{}.npy'.format(layer)

		cond_flist = glob(cond_name)
		condg_flist = glob(condg_name)

		for i in range(len(cond_flist)):
			# load cond_flist
			cond = np.load(cond_flist[i])
			condg = np.load(condg_flist[i])

			cond = cond.reshape(10,-1,cond.shape[-1])
			condg = condg.reshape(10,-1,condg.shape[-1])

			cond_layer.append(cond)
			condg_layer.append(condg)

		cond_layer = np.concatenate(cond_layer)
		condg_layer = np.concatenate(condg_layer)

		cond_VAR_mean_ = cond_layer.var(axis=0).mean(axis=1).mean()
		cond_VAR_std_ = cond_layer.var(axis=0).mean(axis=1).std()
		condg_VAR_mean_ = condg_layer.var(axis=0).mean(axis=1).mean()
		condg_VAR_std_ = condg_layer.var(axis=0).mean(axis=1).std()

		cond_VAR_mean.append(cond_VAR_mean_)
		cond_VAR_std.append(cond_VAR_std_)
		condg_VAR_mean.append(condg_VAR_mean_)
		condg_VAR_std.append(condg_VAR_std_)

		cond_adap_VAR_mean_ = cond_layer.var(axis=2).mean(axis=1).mean()
		cond_adap_VAR_std_ = cond_layer.var(axis=2).mean(axis=1).std()
		condg_adap_VAR_mean_ = condg_layer.var(axis=2).mean(axis=1).mean()
		condg_adap_VAR_std_ = condg_layer.var(axis=2).mean(axis=1).std()

		cond_adap_VAR_mean.append(cond_adap_VAR_mean_)
		cond_adap_VAR_std.append(cond_adap_VAR_std_)
		condg_adap_VAR_mean.append(condg_adap_VAR_mean_)
		condg_adap_VAR_std.append(condg_adap_VAR_std_)


	cond_ = pd.DataFrame({'layer1_mean':cond_VAR_mean[0], 'layer1_std':cond_VAR_std[0],
						  'layer2_mean':cond_VAR_mean[1], 'layer2_std':cond_VAR_std[1],
						  'layer3_mean':cond_VAR_mean[2], 'layer3_std':cond_VAR_std[2]}, index=[0])
	condg_ = pd.DataFrame({'layer1_mean':condg_VAR_mean[0], 'layer1_std':condg_VAR_std[0],
														  'layer2_mean':condg_VAR_mean[1], 'layer2_std':condg_VAR_std[1],
						  'layer3_mean':condg_VAR_mean[2], 'layer3_std':condg_VAR_std[2]}, index=[0])
	# save dataframe to csv
	cond_.to_csv(dir + 'condnet_stability_var_tau' + str(tau) + '.csv')
	condg_.to_csv(dir + 'condgnet_stability_var_tau' + str(tau) + '.csv')

	cond_adap_ = pd.DataFrame({'layer1_mean':cond_adap_VAR_mean[0], 'layer1_std':cond_adap_VAR_std[0],
							   						  'layer2_mean':cond_adap_VAR_mean[1], 'layer2_std':cond_adap_VAR_std[1],
							   						  'layer3_mean':cond_adap_VAR_mean[2], 'layer3_std':cond_adap_VAR_std[2]}, index=[0])
	condg_adap_ = pd.DataFrame({'layer1_mean':condg_adap_VAR_mean[0], 'layer1_std':condg_adap_VAR_std[0],
															   						  'layer2_mean':condg_adap_VAR_mean[1], 'layer2_std':condg_adap_VAR_std[1],
															   						  'layer3_mean':condg_adap_VAR_mean[2], 'layer3_std':condg_adap_VAR_std[2]}, index=[0])

	cond_adap_.to_csv(dir + 'condnet_adap_stability_var_tau' + str(tau) + '.csv')
	condg_adap_.to_csv(dir + 'condgnet_adap_stability_var_tau' + str(tau) + '.csv')






		# # entropy of each unit
		# cond_layer = np.exp(cond_layer)
		# condg_layer = np.exp(condg_layer)
		#
		# cond_layer = cond_layer / cond_layer.sum(axis=0)
		# condg_layer = condg_layer / condg_layer.sum(axis=0)
		#
		# cond_layer.var(axis=0).mean()
		# condg_layer.var(axis=0).mean()

