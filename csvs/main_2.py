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

class cnn(nn.Module):
	def __init__(self):
		super(cnn, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
		self.fc1 = nn.Linear(8 * 156 * 64, 1024)
		self.fc2 = nn.Linear(1024, 2)
		self.bn1 = nn.BatchNorm2d(32)
		self.bn2 = nn.BatchNorm2d(64)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)
		x = F.max_pool2d(x, 2)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.bn2(x)
		x = F.max_pool2d(x, 2)
		x = x.view(-1, 8 * 156 * 64)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.softmax(x, dim=1)
		return x

class cam(nn.Module):
	def __init__(self):
		super(cam, self).__init__()
		self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2, bias=False)
		self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False)
		self.fc1 = nn.Linear(256, 2)
		self.bn1 = nn.BatchNorm2d(128)
		self.bn2 = nn.BatchNorm2d(256)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.bn1(x)
		x = F.max_pool2d(x, 2, padding=1)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.bn2(x)
		x = F.max_pool2d(x, 2, padding=1)

		# global average pooling
		x = torch.mean(x, dim=(2,3))
		x = self.fc1(x)
		x = F.softmax(x, dim=1)
		return x

class mlp(nn.Module):
	def __init__(self, num_input, device):
		super(mlp, self).__init__()
		self.num_input = num_input
		self.layers = nn.ModuleList()
		temp_list = [self.num_input,4096, 1024, 256, 2]
		layers_in_list = temp_list[:-1]
		layers_out_list = temp_list[1:]
		for i in range(len(layers_in_list)):
			self.layers.append(nn.Linear(layers_in_list[i], layers_out_list[i]))
		self.bn_layers = nn.ModuleList()
		for i in range(len(layers_out_list)):
			self.bn_layers.append(nn.BatchNorm1d(layers_out_list[i]))

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)
			x = self.bn_layers[i](x)
			x_ = F.relu(x)
			x = x_
		x = F.softmax(x_, dim=1)
		return x


class mlp_s(nn.Module):
	def __init__(self, num_input, device):
		super(mlp_s, self).__init__()
		self.num_input = num_input
		self.layers = nn.ModuleList()
		temp_list = [self.num_input, 1024, 2]
		layers_in_list = temp_list[:-1]
		layers_out_list = temp_list[1:]
		for i in range(len(layers_in_list)):
			self.layers.append(nn.Linear(layers_in_list[i], layers_out_list[i]))
		self.bn_layers = nn.ModuleList()
		for i in range(len(layers_out_list)):
			self.bn_layers.append(nn.BatchNorm1d(layers_out_list[i]))

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.layers[i](x)
			x = self.bn_layers[i](x)
			x_ = F.relu(x)
			x = x_
		x = F.softmax(x_, dim=1)
		return x

# gnn using pytorch geometric, and DenseSAGEConv
class gnn(nn.Module):
	def __init__(self,device):
		super(gnn, self).__init__()
		self.device =device
		self.conv1 = DenseSAGEConv(626, 626)
		self.conv2 = DenseSAGEConv(626, 626)
		self.conv3 = DenseSAGEConv(626, 626)
		self.fc1 = nn.Linear(32*626, 1024)
		self.fc2 = nn.Linear(1024, 2)

	def forward(self, x):
		adj = torch.ones((x.size(0),32,32)).to(self.device)
		x = self.conv1(x, adj)
		x = F.tanh(x)
		x = self.conv2(x, adj)
		x = F.tanh(x)
		x = self.conv3(x, adj)
		x = F.tanh(x)
		x = x.view(-1,32*626)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.softmax(x, dim=1)
		return x



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
	args.add_argument('--model', type=str, default='mlp_s')
	# args.add_argument('--model', type=str, default='cnn')
	args.add_argument('--mode', type=str, default='ECEO')
	args.add_argument('--epoch', type=int, default=300)
	args.add_argument('--learning_rate', type=float, default=3e-5)
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


	# device name
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if mode != 2:
		# load data
		data, label = load_data(mode)

	elif mode ==2:
		data, label = load_data(0)
		data_, label_ = load_data(1)
		data = np.concatenate((data, data_), axis=0)
		label = np.concatenate((label, label_), axis=0)

		# mix them again
		idx = np.random.permutation(data.shape[0])
		data = data[idx]
		label = label[idx]

	print('data loaded')

	if args.model == 'mlp':
		# make flatten data
		data = data.reshape(data.shape[0], -1)
		model = mlp(data.shape[1], device).to(device)
	if args.model == 'mlp_s':
		# make flatten data
		data = data.reshape(data.shape[0], -1)
		model = mlp_s(data.shape[1], device).to(device)
	elif args.model == 'cnn':
		# make flatten data
		data = data.reshape(data.shape[0],1,data.shape[1],data.shape[2])
		model = cnn().to(device)
	elif args.model == 'cam':
		# make flatten data
		data = data.reshape(data.shape[0],1,data.shape[1],data.shape[2])
		model = cam().to(device)
	elif args.model == 'gnn':
		# make flatten data
		model = gnn(device).to(device)

	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	# print number of parameters
	print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")

	wandb.init(project="adhd-tic_real",
				config=args.__dict__,
			   name=args.model + '_' + args.mode + '_' + str(args.learning_rate) + '_' + str(args.batchsize))

	wandb.log({'num_params': num_params}, step=0)

	# make training part
	train_data = data[:int(data.shape[0]*0.8)]
	train_label = label[:int(data.shape[0]*0.8)]
	# make test part
	test_data = data[int(data.shape[0]*0.8):]
	test_label = label[int(data.shape[0]*0.8):]
	test_data = torch.Tensor(test_data).float().to(device)
	test_label = torch.Tensor(test_label).long().to(device)
	# make onehot vector for test_label

	# make batch
	batch_size = args.batchsize
	num_batch = int(train_data.shape[0]/batch_size)

	# initialize model and optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()
	model.train()

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999)

	# make batch training code
	for epoch in range(args.epoch):
		log_train = {}
		model.train()
		loss_list = []
		accs = []
		for i in range(num_batch):
			batch_data = train_data[i*batch_size:(i+1)*batch_size]
			batch_label = train_label[i*batch_size:(i+1)*batch_size]

			# make torch tensor
			batch_data = torch.Tensor(batch_data).to(device)
			batch_label = torch.Tensor(batch_label).to(device)


			optimizer.zero_grad()
			# make prediction
			pred = model(batch_data)
			# calculate loss
			loss = criterion(pred, batch_label.long())
			# backpropagation
			loss.backward()
			optimizer.step()

			# argmax for pred
			pred_c = torch.argmax(pred, dim=1)

			# append loss to loss_list
			loss_list.append(loss.item())
			# get accuracy
			acc = torch.sum(pred_c == batch_label) / batch_label.shape[0]
			accs.append(acc.item())

			# wandb.log({'train/batch/loss': loss.item(), 'train/batch/acc': acc})

			# print loss and accuracy
			if (i+1) % 5 == 0:
				print('epoch: %d, batch: %d, loss: %.3f, acc: %.3f' % (epoch, i+1, loss.item(), acc))

		scheduler.step()
		wandb.log({'train/epoch/loss': np.mean(loss_list), 'train/epoch/acc': np.mean(accs)}, step=epoch)

		with torch.no_grad():
			pred = model(test_data)

			loss = criterion(pred, test_label.long())
			pred_c = torch.argmax(pred, dim=1)
			acc = torch.sum(pred_c == test_label) / test_label.shape[0]
		print('epoch: %d, loss: %.7f, test acc: %.3f' % (epoch+1, np.mean(loss_list), acc))
		wandb.log({'test/epoch/loss': loss.item(), 'test/epoch/acc': acc},step=epoch)

		# save model
		torch.save(model.state_dict(), 'model/' + args.model + '_' + args.mode + '_' + str(args.learning_rate) + '_' + str(args.batchsize) + '.pt')


	wandb.finish()

if __name__ == '__main__':
	main()
