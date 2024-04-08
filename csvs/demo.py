# code that is for checking variance in the data

import scipy.io as sio
from glob import glob
import numpy as np
from scipy import signal
from torch import nn
from torch.functional import F
import torch
import wandb
import torch_geometric
from torch_geometric.nn import DenseSAGEConv

seed_num = 2121
_mode_run = ['EC', 'EO']

def load_data(mode):
	if mode== 0:
		cond = 'EC'
	elif mode == 1:
		cond = 'EO'
	else:
		raise ValueError('mode should be 0 or 1')

	flist = glob('./dat_sub/*_' + cond + '.mat')
	# load mat from ./dat_sub
	for f in flist:
		if 'ADHD' in f:
			adhd = sio.loadmat(f)['epoch']
		elif 'TIC' in f:
			tic  = sio.loadmat(f)['epoch']
		else:
			raise ValueError('file name should contain ADHD or TIC')

	# extract power spectrum of each channel from adhd

	fa, adhd_ = signal.periodogram(adhd, fs=250, axis=1)
	ft, tic_  = signal.periodogram(tic, fs=250, axis=1)

	# set random seed
	np.random.seed(seed_num)

	# mix adhd
	adhd_idx = np.random.permutation(adhd_.shape[2])
	adhd_ = adhd_[:,:,adhd_idx]

	# mix tic
	tic_idx = np.random.permutation(tic_.shape[2])
	tic_ = tic_[:,:,tic_idx]

	# cut with minimum length for both
	min_len = min(adhd_.shape[2], tic_.shape[2])
	adhd_ = adhd_[:,:,:min_len]
	tic_  = tic_[:,:,:min_len]

	# merge them into one array data and make label
	data = np.concatenate((adhd_, tic_), axis=2)

	# swapaxes to make data shape (trials, channel, time)
	data = np.swapaxes(data, 1, 2)
	data = np.swapaxes(data, 0, 1)

	label = np.concatenate((np.zeros(adhd_.shape[2]), np.ones(tic_.shape[2])), axis=0)

	# mix them again
	idx = np.random.permutation(data.shape[0])
	data = data[idx]
	label = label[idx]

	return data, label


def main():
	import argparse
	args = argparse.ArgumentParser()
	# args.add_argument('--model', type=str, default='cnn')
	args.add_argument('--mode', type=str, default='ECEO')
	args.add_argument('--epoch', type=int, default=300)
	args.add_argument('--learning_rate', type=float, default=7e-4)
	args.add_argument('--batchsize', type=int, default=100)

	args = args.parse_args()

	if args.mode == 'EC':
		mode = 0
	elif args.mode == 'EO':
		mode = 1
	elif args.mode == 'ECEO':
		mode = 2
	else:
		raise ValueError('mode should be EC or EO')

	# load data, and extract power spectrum from each channel. see line 37-38 using "periodogram"
	if mode != 2:
		data, label = load_data(mode)
	elif mode ==2:
		data, label = load_data(0)
		data_, label_ = load_data(1)
		data = np.concatenate((data, data_), axis=0)
		label = np.concatenate((label, label_), axis=0)

		# mix them again for reproducibility
		idx = np.random.permutation(data.shape[0])
		data = data[idx]
		label = label[idx]

	print('data loaded')

	# print data shape
	print(data.shape) # (number of data, channels, frequency)


if __name__ == '__main__':
	main()
