import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import pandas as pd

from glob import glob

from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Mlp(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.ModuleList()
		self.layers.append(nn.Linear(28*28, 512))
		self.layers.append(nn.Linear(512, 256))
		self.layers.append(nn.Linear(256, 10))

	def forward(self, x, cond_drop=False, us=None):
		hs = [x]
		# flatten
		len_in = 0
		len_out = x.shape[1]
		if not cond_drop:
			for layer in self.layers:
				x = layer(x)
				x = F.sigmoid(x)
				# dropout
				# x = nn.Dropout(p=0.3)(x)
				hs.append(x)
		else:
			if us is None:
				raise ValueError('u should be given')
			# conditional activation
			for layer in self.layers:
				us = us.squeeze()
				len_out = layer.in_features
				x = x * us[:,len_in:len_in+len_out] # where it cuts off [TODO]
				x = layer(x)
				x = F.relu(x)
				# dropout
				# x = nn.Dropout(p=0.3)(x)
				len_in = len_out
				hs.append(x)

		# softmax
		x = F.softmax(x, dim=1)
		return x, hs

class Gnn(nn.Module):
	def __init__(self, minprob, maxprob, hidden_size = 64):
		super().__init__()
		self.conv1 = DenseSAGEConv(1, hidden_size)
		self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
		self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
		self.fc1 = nn.Linear(hidden_size, 1)
		# self.fc2 = nn.Linear(64,1, bias=False)
		self.minprob = minprob
		self.maxprob = maxprob

	def forward(self, hs, adj):
		# hs : hidden activity
		batch_adj = torch.stack([torch.Tensor(adj) for _ in range(hs.shape[0])])
		batch_adj = batch_adj.to(device)

		hs_0 = hs.unsqueeze(-1)

		hs = F.sigmoid(self.conv1(hs_0, batch_adj))
		hs = F.sigmoid(self.conv2(hs, batch_adj))
		hs = F.sigmoid(self.conv3(hs, batch_adj))
		hs_conv = hs
		hs = self.fc1(hs)
		# hs = self.fc2(hs)
		p = F.sigmoid(hs)
		# bernoulli sampling
		p = p * (self.maxprob - self.minprob) + self.minprob
		u = torch.bernoulli(p).to(device) # [TODO] Error : probability -> not a number(nan), p is not in range of 0 to 1

		return u, p


class model_condnet(nn.Module):
	def __init__(self,args):
		super().__init__()
		if torch.cuda.is_available():
			self.device = 'cuda'
		else:
			self.device = 'cpu'

		self.input_dim = 28*28
		mlp_hidden = [512, 256, 10]
		output_dim = mlp_hidden[-1]

		nlayers = args.nlayers
		self.condnet_min_prob = args.condnet_min_prob
		self.condnet_max_prob = args.condnet_max_prob

		self.mlp_nlayer = 0

		self.mlp = nn.ModuleList()
		self.mlp.append(nn.Linear(self.input_dim, mlp_hidden[0]))
		for i in range(nlayers):
			self.mlp.append(nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
		self.mlp.append(nn.Linear(mlp_hidden[i+1], output_dim))
		self.mlp.to(self.device)

		#DOWNSAMPLE
		self.avg_poolings = nn.ModuleList()
		pool_hiddens = [512, *mlp_hidden]
		for i in range(len(self.mlp)):
			stride = round(pool_hiddens[i] / pool_hiddens[i+1])
			self.avg_poolings.append(nn.AvgPool1d(kernel_size=stride, stride=stride))

		#UPSAMPLE
		self.upsample = nn.ModuleList()
		for i in range(len(self.mlp)):
			stride = round(pool_hiddens[i+1] / 1024)
			self.upsample.append(nn.Upsample(scale_factor=stride, mode='nearest'))


		# HANDCRAFTED POLICY NET
		n_each_policylayer = 1
		# n_each_policylayer = 1 # if you have only 1 layer perceptron for policy net
		self.policy_net = nn.ModuleList()
		temp = nn.ModuleList()
		# temp.append(nn.Linear(self.input_dim, mlp_hidden[0])) # BEFORE LARGE MODEL'S
		temp.append(nn.Linear(self.input_dim, 1024))
		self.policy_net.append(temp)

		for i in range(len(self.mlp)-1):
			temp = nn.ModuleList()
			for j in range(n_each_policylayer):
				# temp.append(nn.Linear(self.mlp[i].out_features, self.mlp[i].out_features)) # BEFORE LARGE MODEL'S
				temp.append(nn.Linear(self.mlp[i].out_features, 1024))
			self.policy_net.append(temp)
		self.policy_net.to(self.device)

	def forward(self, x):
		# return policies
		policies = []
		sample_probs = []
		layer_masks = []

		x = x.view(-1, self.input_dim).to(self.device)

		# for each layer
		h = x
		u = torch.ones(h.shape[0], h.shape[1]).to(self.device)

		for i in range(len(self.mlp)-1):
			# h_clone = h.clone()
			# p_i = self.policy_net[i][0](h_clone.detach())
			p_i = self.policy_net[i][0](h)

			p_i = F.sigmoid(p_i)
			for j in range(1, len(self.policy_net[i])):
				p_i = self.policy_net[i][j](p_i)
				p_i = F.sigmoid(p_i)

			p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
			u_i = torch.bernoulli(p_i).to(self.device)

			# debug[TODO]
			# u_i = torch.ones(u_i.shape[0], u_i.shape[1])

			if u_i.sum() == 0:
				idx = np.random.uniform(0, u_i.shape[0], size = (1)).astype(np.int16)
				u_i[idx] = 1

			sampling_prob = p_i * u_i + (1-p_i) * (1-u_i)

			# idx = torch.where(u_i == 0)[0]

			# h_next = F.relu(self.mlp[i](h*u.detach()))*u_i

			# compresss u_i to size of u

			# WHEN YOU DO DOWNSAMPLE
			# u_i = self.avg_poolings[i](u_i)

			# WHEN YOU DO UPSAMPLE
			# u_i = self.upsample[i](u_i.unsqueeze(0)).squeeze(0)
			u_i = F.interpolate(u_i.unsqueeze(1), size=self.mlp[i].out_features, mode='linear', align_corners=True).squeeze(1)

			h_next = F.relu(self.mlp[i](h*u))*u_i
			h = h_next
			u = u_i

			policies.append(p_i)
			sample_probs.append(sampling_prob)
			layer_masks.append(u_i)

		p_i = self.policy_net[-1][0](h)

		p_i = F.sigmoid(p_i)
		for j in range(1, len(self.policy_net[-1])):
			p_i = self.policy_net[i][j](p_i)
			p_i = F.sigmoid(p_i)

		p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
		u_i = torch.bernoulli(p_i).to(self.device)

		if u_i.sum() == 0:
			idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
			u_i[idx] = 1


		sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
		u_i = F.interpolate(u_i.unsqueeze(1), size=10, mode='linear', align_corners=True).squeeze(1)


		h = self.mlp[-1](h*u) * u_i

		sample_probs.append(sampling_prob)

		layer_masks.append(u_i)

		# last layer just go without dynamic sampling
		# h = self.mlp[-1](h)
		h = F.softmax(h, dim=1)

		return h, policies, sample_probs, layer_masks

def adj(model, bidirect = True, last_layer = True, edge2itself = True):
	if last_layer:
		num_nodes = sum([layer.in_features for layer in model.layers]) + model.layers[-1].out_features
		nl = len(model.layers)
		trainable_nodes = np.concatenate(
			(np.ones(sum([layer.in_features for layer in model.layers])), np.zeros(model.layers[-1].out_features)),
			axis=0)
		# trainable_nodes => [1,1,1,......,1,0,0,0] => input layer & hidden layer 의 노드 개수 = 1의 개수, output layer 의 노드 개수 = 0의 개수
	else:
		num_nodes = sum([layer.in_features for layer in model.layers])
		nl = len(model.layers) - 1
		trainable_nodes = np.ones(num_nodes)

	adjmatrix = np.zeros((num_nodes, num_nodes), dtype=np.int16)
	current_node = 0

	for i in range(nl):
		layer = model.layers[i]
		num_current = layer.in_features
		num_next = layer.out_features

		for j in range(current_node, current_node + num_current):
			for k in range(current_node + num_current, current_node + num_current + num_next):
				adjmatrix[j, k] = 1
		# print start and end for j
		print(current_node, current_node + num_current)
		# print start and end for k
		print(current_node + num_current, current_node + num_current + num_next)
		current_node += num_current

	if bidirect:
		adjmatrix += adjmatrix.T

	if edge2itself:
		adjmatrix += np.eye(num_nodes, dtype=np.int16)
		# make sure every element that is non-zero is 1
	adjmatrix[adjmatrix != 0] = 1
	return adjmatrix, trainable_nodes
def main(args):
	# get args
	lambda_s = args.lambda_s
	lambda_v = args.lambda_v
	lambda_l2 = args.lambda_l2
	lambda_pg = args.lambda_pg
	tau = args.tau
	learning_rate = args.lr
	max_epochs = args.max_epochs
	BATCH_SIZE = args.BATCH_SIZE


	test_dataset = datasets.MNIST(
		root="../data/mnist",
		train=False,
		download=True,
		transform=transforms.ToTensor()
	)

	test_loader = torch.utils.data.DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False
	)


	for iter in range(10):
		for tau in [0.3, 0.6, 0.9]:
			condnet_stability_df = pd.DataFrame(columns=['layer1', 'layer2', 'layer3'])
			condgnet_stability_df = pd.DataFrame(columns=['layer1', 'layer2', 'layer3'])

			# get list of models using glob
			condnet_list = glob('./condnet_lastlayer/{}/cond1024*.pt'.format(tau))
			congnet_gnn_list = glob('./condgpt/{}/gnn_policy*.pt'.format(tau))
			congnet_mlp_list = glob('./condgpt/{}/mlp_model_s*.pt'.format(tau))

			condnet_iter = [[],[],[]]
			condgnet_iter = [[],[],[]]

			for i in range(len(condnet_list)):
				# create model
				model = model_condnet(args)
				# load model from condnet_list[i]
				model.load_state_dict(torch.load(condnet_list[i],map_location=torch.device('cpu')))

				mlp_model = Mlp().to(device)
				gnn_policy = Gnn(minprob=args.condnet_min_prob, maxprob=args.condnet_max_prob, hidden_size=128).to(device)
				# load model from congnet_gnn_list[i] and congnet_mlp_list[i]
				mlp_model.load_state_dict(torch.load(congnet_mlp_list[i],map_location=torch.device('cpu')))
				gnn_policy.load_state_dict(torch.load(congnet_gnn_list[i],map_location=torch.device('cpu')))

				mlp_surrogate = Mlp().to(device)
				# copy weights in mlp to mlp_surrogate
				mlp_surrogate.load_state_dict(mlp_model.state_dict())
				adj_, nodes_ = adj(mlp_model)

				# make empty data frame to save csv


				model.eval()
				gnn_policy.eval()
				mlp_model.eval()
				with torch.no_grad():
					# calculate accuracy on test set
					acc = 0
					bn = 0


					condnet_tau_batch = [[],[],[]]
					condgnet_tau_batch = [[],[],[]]

					condnet_batch = [[],[],[]]
					condgnet_batch = [[],[],[]]

					for i, data in enumerate(test_loader, 0):
						bn += 1
						# get batch
						inputs, labels = data

						# make one hot vector
						y_batch_one_hot = torch.zeros(labels.shape[0], 10)
						y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1,).tolist()] = 1

						# get output
						outputs, policies, sample_probs, layer_masks = model(torch.tensor(inputs))

						mlp_surrogate.eval()
						inputs = inputs.view(-1, 28**2).to(device)
						outputs_1, hs = mlp_surrogate(inputs)
						hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
									   dim=1)  # changing dimension to 1 for putting hs vector in gnn
						hs = hs.detach()
						us, p = gnn_policy(hs, adj_)  # run gnn
						outputs, hs = mlp_model(inputs, cond_drop=True, us=us.detach())

						uss = [us[:, 784:1296].squeeze(),us[:, 1296:1552].squeeze(),us[:, 1552:].squeeze()]

						taus_condg = [p[:, 784:1296].mean().item(), p[:, 1296:1552].mean().item(), p[:, 1552:].mean().item()]
						taus_cond = [l.mean().item() for l in sample_probs]

						sample_probs = [np.round(l.to('cpu').detach()) for l in layer_masks]

						uss = [l.to('cpu').detach() for l in uss]

						for i in range(3):
							condnet_iter[i].append(sample_probs[i].numpy())
							condgnet_iter[i].append(uss[i].numpy())


			condnet_iter = [np.concatenate(L,axis=0) for L in condnet_iter]
			condgnet_iter = [np.concatenate(L,axis=0) for L in condgnet_iter]

			for i in range(3):
				np.save('./condnet_iter{}_tau{}_layer{}.npy'.format(iter,tau,i), condnet_iter[i])
				np.save('./condgnet_iter{}_tau{}_layer{}.npy'.format(iter,tau,i), condgnet_iter[i])











if __name__=='__main__':
	# make arguments and defaults for the parameters
	import argparse
	args = argparse.ArgumentParser()
	args.add_argument('--nlayers', type=int, default=1)
	args.add_argument('--lambda_s', type=float, default=7)
	args.add_argument('--lambda_v', type=float, default=1.2)
	args.add_argument('--lambda_l2', type=float, default=5e-4)
	args.add_argument('--lambda_pg', type=float, default=1e-3)
	args.add_argument('--tau', type=float, default=0.2)
	args.add_argument('--lr', type=float, default=0.1)
	args.add_argument('--max_epochs', type=int, default=1000)
	args.add_argument('--condnet_min_prob', type=float, default=0.1)
	args.add_argument('--condnet_max_prob', type=float, default=0.9)
	args.add_argument('--learning_rate', type=float, default=0.1)
	args.add_argument('--BATCH_SIZE', type=int, default=256)

	# get time in string to save as file name
	now = datetime.now()
	dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

	main(args=args.parse_args())
