import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

from tqdm import tqdm
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
        len_in = 0
        len_out = x.shape[1]
        if not cond_drop:
            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
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
    def __init__(self, minprob, maxprob):
        super().__init__()
        self.conv1 = DenseSAGEConv(1, 128)
        self.conv2 = DenseSAGEConv(128, 128)
        self.conv3 = DenseSAGEConv(128, 128)
        self.fc1 = nn.Linear(128, 1)
        self.minprob = minprob
        self.maxprob = maxprob

    def forward(self, hs, adj):
        # hs : hidden activity
        batch_adj = torch.stack([torch.Tensor(adj) for _ in range(hs.shape[0])])
        batch_adj = batch_adj.to(device)

        hs = F.relu(self.conv1(hs.unsqueeze(-1), batch_adj))
        hs = F.relu(self.conv2(hs, batch_adj))
        hs = F.relu(self.conv3(hs, batch_adj))
        hs = self.fc1(hs)
        p = F.sigmoid(hs)
        # bernoulli sampling
        p = p * (self.maxprob - self.minprob) + self.minprob
        u = torch.bernoulli(p).to(device) # [TODO] Error : probability -> not a number(nan), p is not in range of 0 to 1
        # x = p * u + (1 - p) * (1 - u) # 논문에서는 각 레이어마다 policy가 적용되었기때문인데, 온오프 노드를 만들거기 때문에 이 연산은 필요없을 듯...
        return u, p

class Condnet_model(nn.Module):
    def __init__(self, args, num_input = 28**2):
        super().__init__()
        # get args
        self.lambda_s = args.lambda_s
        self.lambda_v = args.lambda_v
        self.lambda_l2 = args.lambda_l2
        self.lambda_pg = args.lambda_pg
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.max_epochs = args.max_epochs
        self.BATCH_SIZE = args.BATCH_SIZE
        self.num_input = num_input
        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob
        self.compact = args.compact
        self.mlp = Mlp().to(device)
        self.gnn = Gnn(minprob = self.condnet_min_prob, maxprob = self.condnet_max_prob).to(device)
        self.mlp_surrogate = Mlp().to(device)
        # copy weights in mlp to mlp_surrogate
        self.mlp_surrogate.load_state_dict(self.mlp.state_dict())

        self.C = nn.CrossEntropyLoss()

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

    # 1. inference로 돌리는 함수 (mlp 모든 unit이 1로 켜져있는 경우 (이때 업데이트는 의미없음) ) => model.evaluation
    def infer_forward_propagation(self, inputs, adj):
        # self.mlp.eval()
        # self.gnn.eval()
        # run the first mlp
        y_pred, hs = self.mlp(inputs)
        hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
                       dim=1)  # changing dimension to 1 for putting hs vector in gnn

        return y_pred, hs

    # 2. train으로 돌리는 함수
    def forward_propagation(self, inputs, adj, hs):
        self.mlp.train()
        self.gnn.train()

        us, p = self.gnn(hs, adj)  # run gnn
        outputs, hs = self.mlp(inputs, cond_drop=True, us=us.detach())

        return outputs, us, hs, p

    def compute_cost(self, y_pred, labels, us, p):

        c = nn.CrossEntropyLoss()(y_pred, labels.to(device))
        # c /= y_pred.shape[0]

        #
        Ls = self.lambda_s * (
                torch.pow(us.mean(axis=0) - torch.tensor(self.tau).to(device), 2).mean() +
                torch.pow(us.mean(axis=1) - torch.tensor(self.tau).to(device), 2).mean())

        # Compute the regularization loss L
        L = c + Ls
        L += self.lambda_v * (-1) * (us.to(device).var(axis=0).mean() +
                                     us.to(device).var(axis=1).mean())

        # Compute the policy gradient (PG) loss
        logp = torch.log(p).sum(axis=1).mean()

        PG = self.lambda_pg * c * (-logp) + L

        # PG /= y_pred.shape[0]

        return c, PG

    # def update_parameters(self, optimizer_mlp, optimizer_gnn, PG):

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

        optimizer_mlp = optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        optimizer_gnn = optim.Adam(self.gnn.parameters(), lr=self.learning_rate)

        adj_, nodes_ = self.adj(self.mlp)

        for epoch in range(self.max_epochs):

            self.mlp.train()
            self.gnn.train()
            costs = 0
            accs = 0
            accsbf = 0
            PGs = 0
            num_iteration = 0
            us = torch.zeros((1562, 1562))

            for i, data in enumerate(tqdm(train_loader), start=0):

                if self.compact:
                    if i>50:
                        break

                optimizer_mlp.zero_grad()
                optimizer_gnn.zero_grad()

                num_iteration += 1

                # get batch
                inputs, labels = data
                inputs = inputs.view(-1, self.num_input).to(device)

                # Forward Propagation
                # ouputs, hs     = self.infer_forward_propagation(inputs, adj_)
                # y_pred, us, hs, p = self.forward_propagation(inputs, adj_, hs.detach())
                self.mlp_surrogate.eval()
                ouputs, hs = self.mlp_surrogate(inputs)
                hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
                               dim=1)  # changing dimension to 1 for putting hs vector in gnn
                hs = hs.detach()

                # us, p = self.gnn(hs, adj_)  # run gnn
                y_pred, hs = self.mlp(inputs)

                # Compute Cost
                # c, PG = self.compute_cost(y_pred, labels, us, p)

                c = self.C(y_pred, labels.to(device))

                # Ls = self.lambda_s * (
                #         torch.pow(us.mean(axis=0) - torch.tensor(self.tau).to(device), 2).mean() +
                #         torch.pow(us.mean(axis=1) - torch.tensor(self.tau).to(device), 2).mean())
                # # Compute the regularization loss L
                # L = c + Ls
                # L += self.lambda_v * (-1) * (us.to(device).var(axis=0).mean() +
                #                              us.to(device).var(axis=1).mean())
                # # Compute the policy gradient (PG) loss
                # logp = torch.log(p).sum(axis=1).mean()
                # PG = self.lambda_pg * c * (-logp) + L

                # Backward Optimization
                c.backward()  # it needs to be checked [TODO]
                optimizer_mlp.step()
                # optimizer_gnn.step()

                # update surrogate
                self.mlp_surrogate.load_state_dict(self.mlp.state_dict())

                costs += c.to('cpu').item()
                # PGs += PG.to('cpu').item()


                # Calculate accuracy
                y_pred = torch.argmax(y_pred.to('cpu'), dim=1)  # why should it be used??
                acc = torch.sum(y_pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
                accs += acc

                ouputs = torch.argmax(ouputs.to('cpu'), dim=1)
                acc_ = torch.sum(ouputs==torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
                accsbf += acc_
                if (i % 50) == 0:
                    # Print PG.item(), and acc with name
                    # print('\nEpoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}, Accbf {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, acc_))
                    print('\nEpoch: {}, Batch: {}, Cost: {:.10f}, Acc: {:.3f}, Accbf {:.3f}'.format(epoch, i, c.item(),  acc, acc_))

                    # print how may are turned
                    print('TAU: {}'.format(us.cpu().mean().item()))


            # Print epoch and epochs costs and accs
            print('Epoch: {}, Cost: {}, Accuracy: {}, Accuracy_bf: {}'.format(epoch, costs / num_iteration, accs / num_iteration, accsbf / num_iteration))

if __name__=='__main__':
    # make arguments and defaults for the parameters
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=20)
    args.add_argument('--lambda_v', type=float, default=2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-1)
    args.add_argument('--tau', type=float, default=0.5)
    args.add_argument('--max_epochs', type=int, default=1000)
    args.add_argument('--condnet_min_prob', type=float, default=1)
    args.add_argument('--condnet_max_prob', type=float, default=1)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=512)
    args.add_argument('--compact', type=bool, default=True)

    model = Condnet_model(args=args.parse_args())
    model.run_model()


