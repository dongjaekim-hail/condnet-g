import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

import torch.nn as nn
import torch.nn.functional as F
# import wandb

from datetime import datetime

from torch_geometric.nn import DenseSAGEConv

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
        if not cond_drop:
            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
                # dropout
                x = nn.Dropout(p=0.3)(x)
                hs.append(x)
        else:
            if us is None:
                raise ValueError('u should be given')
            # conditional activation
            for layer in self.layers:
                us = us.squeeze()
                x = x * us[:,:layer.in_features] # where it cuts off [TODO]
                x = layer(x) 
                x = F.relu(x)
                # dropout
                x = nn.Dropout(p=0.3)(x)
                hs.append(x)

        # softmax
        x = F.softmax(x, dim=1)
        return x, hs

class Gnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DenseSAGEConv(1, 128)
        self.conv2 = DenseSAGEConv(128, 128)
        self.conv3 = DenseSAGEConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, hs, adj):
        # hs : hidden activity
        batch_adj = torch.stack([torch.Tensor(adj) for _ in range(hs.shape[0])])

        hs = F.relu(self.conv1(hs.unsqueeze(-1), batch_adj))
        hs = F.relu(self.conv2(hs, batch_adj))
        hs = F.relu(self.conv3(hs, batch_adj))
        hs = self.fc1(hs)
        p = F.sigmoid(hs)
        # bernoulli sampling
        p = p * (0.7 - 0.1) + 0.1
        u = torch.bernoulli(p).to(device) # [TODO] Error : probability -> not a number(nan), p is not in range of 0 to 1
        # x = p * u + (1 - p) * (1 - u) # 논문에서는 각 레이어마다 policy가 적용되었기때문인데, 온오프 노드를 만들거기 때문에 이 연산은 필요없을 듯...
        return u

class Condnet_model(nn.Module):
    def __init__(self, args, num_input = 28**2):
        super().__init__()
        # get args
        self.lambda_s = args.lambda_s
        self.lambda_v = args.lambda_v
        self.lambda_l2 = args.lambda_l2
        self.lambda_pg = args.lambda_pg
        self.tau = args.tau
        self.learning_rate = args.lr
        self.max_epochs = args.max_epochs
        self.BATCH_SIZE = args.BATCH_SIZE
        self.num_input = num_input
        self.mlp = Mlp().to(device)
        self.gnn = Gnn().to(device)

    def adj(self, model, bidirect = True, last_layer = True, edge2itself = True):
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

    def forward_propagation(self, inputs, adj, num_iteration, us):
        if num_iteration == 1:
            # run the first mlp
            y_pred, hs = self.mlp(inputs)
            hs = torch.cat(tuple(hs[i] for i in range(len(hs))), dim=1)  # changing dimension to 1 for putting hs vector in gnn
            us = self.gnn(hs, adj)  # run the gnn
        else:
            # run the second mlp
            y_pred, hs = self.mlp(inputs, cond_drop=True, us=us)
            hs = torch.cat(tuple(hs[i] for i in range(len(hs))), dim=1)  # changing dimension to 1 for putting hs vector in gnn
            us = self.gnn(hs, adj)  # run the gnn

        return y_pred, us, hs

    def compute_cost(self, y_pred, labels, hs):
        y_pred = torch.argmax(y_pred.to('cpu'), dim=1)  # why should it be used??

        # Compute the loss
        c = nn.CrossEntropyLoss()(y_pred.float(), labels.to(device).float())

        # Compute the regularization loss L
        L = c + self.lambda_s * (
                torch.pow(hs.mean(axis=0) - torch.tensor(self.tau).to(device), 2).mean() +
                torch.pow(hs.mean(axis=1) - torch.tensor(self.tau).to(device), 2).mean())
        L += self.lambda_v * (-1) * (hs.to('cpu').var(axis=0).mean() +
                                     hs.to('cpu').var(axis=1).mean())

        # Compute the policy gradient (PG) loss
        logp = torch.log(hs).sum(axis=1).mean()
        PG = self.lambda_pg * c * (-logp) + L

        return c, PG

    def update_parameters(self, optimizer_mlp, optimizer_gnn, PG):
        PG.backward()  # it needs to be checked [TODO]
        optimizer_mlp.step()
        optimizer_gnn.step()

    def run_model(self):
        # datasets load mnist data
        train_dataset = datasets.MNIST(
            root="../data/mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        test_dataset = datasets.MNIST(
            root="../data/mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False
        )

        optimizer_mlp = optim.Adam(self.mlp.parameters(), lr=0.001)
        optimizer_gnn = optim.Adam(self.gnn.parameters(), lr=0.001)

        adj_, nodes_ = self.adj(self.mlp)

        for epoch in range(self.max_epochs):

            self.mlp.train()
            self.gnn.train()
            costs = 0
            accs = 0
            PGs = 0
            num_iteration = 0
            us = torch.zeros((1562, 1562))

            for i, data in enumerate(train_loader, start=0):
                optimizer_mlp.zero_grad()
                optimizer_gnn.zero_grad()

                num_iteration += 1

                # get batch
                inputs, labels = data
                inputs = inputs.view(-1, self.num_input).to(device)

                # Forward Propagation
                us_prev = us
                y_pred, us, hs = self.forward_propagation(inputs, adj_, num_iteration, us_prev)

                # Compute Cost
                c, PG = self.compute_cost(y_pred, labels, hs)
                costs += c.to('cpu').item()
                PGs += PG.to('cpu').item()

                # Backward Optimization
                self.update_parameters(optimizer_mlp, optimizer_gnn, PG)

                # Calculate accuracy
                y_pred = torch.argmax(y_pred.to('cpu'), dim=1)  # why should it be used??
                acc = torch.sum(y_pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # Print PG.item(), and acc with name
                print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc))

            # Print epoch and epochs costs and accs
            print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / num_iteration, accs / num_iteration))

if __name__=='__main__':
    # make arguments and defaults for the parameters
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=20)
    args.add_argument('--lambda_v', type=float, default=2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.2)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--max_epochs', type=int, default=1000)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.7)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=3)

    model = Condnet_model(args=args.parse_args())
    model.run_model()


