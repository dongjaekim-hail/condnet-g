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

# bn 1 to n : data 같은 것에 대해서 돌아야 됨
# mlp us.shape => (1562, 1)으로 / gnn에서 fc( ,1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.condnet_min_prob = 0.1
        self.condnet_max_prob = 0.7

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
                x = x[0] * us[0,:,:layer.in_features] # where it cuts off [TODO]
                x = layer(x) # ??? 각 노드 간의 연결을 끊거나 살리는 것을 us matrix 로 어떻게 연산....
                x = F.relu(x)
                # dropout
                x = nn.Dropout(p=0.3)(x)
                hs.append(x)

        # softmax
        x = F.softmax(x, dim=1)
        return x, hs

class gnn(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.conv1 = DenseSAGEConv(1, 128)
        self.conv2 = DenseSAGEConv(128, 128)
        self.conv3 = DenseSAGEConv(128, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x, edge_index):
        print(x.shape) # hidden activity shape
        batch_adj = torch.stack([edge_index for _ in range(x.shape[0])])

        x = F.relu(self.conv1(x.unsqueeze(-1), batch_adj))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.fc1(x)
        p = F.sigmoid(x)
        # bernoulli sampling
        # p = p * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u = torch.bernoulli(p).to(device)
        # x = p * u + (1 - p) * (1 - u) # 논문에서는 각 레이어마다 policy가 적용되었기때문인데, 온오프 노드를 만들거기 때문에 이 연산은 필요없을 듯...
        return u

class condg_exp():
    def __init__(self, args, num_input = 28**2):
        self.num_input = num_input
        self.mlp = Net().to(device)
        # get args
        self.lambda_s = args.lambda_s
        self.lambda_v = args.lambda_v
        self.lambda_l2 = args.lambda_l2
        self.lambda_pg = args.lambda_pg
        self.tau = args.tau
        self.learning_rate = args.lr
        self.max_epochs = args.max_epochs
        self.BATCH_SIZE = args.BATCH_SIZE

        def adj(model, bidirect = True, last_layer = True, edge2itself = True):

            if last_layer:
                num_nodes = sum([layer.in_features for layer in model.layers]) + model.layers[-1].out_features
                nl = len(model.layers)
                trainable_nodes = np.concatenate((np.ones(sum([layer.in_features for layer in model.layers])), np.zeros(model.layers[-1].out_features)), axis=0)
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
                #make sure every element that is non-zero is 1
            adjmatrix[adjmatrix != 0] = 1
            return adjmatrix, trainable_nodes

        adj_, nodes_ = adj(self.mlp, num_input)
        print(nodes_.sum())
        self.gnn = gnn(nodes_.sum()).to(device)
        self.adj = adj_
        self.optimizer_mlp = optim.Adam(self.mlp.parameters(), lr=0.001)
        self.optimizer_gnn = optim.Adam(self.gnn.parameters(), lr=0.001)

    def update_pol(self, loss, loss_pol):
        self.optimizer_mlp.zero_grad()
        self.optimizer_gnn.zero_grad()

        loss_pol.backward()  # it needs to be checked [TODO]
        self.optimizer_mlp.step()
        self.optimizer_gnn.step()

    def run_exp(self):

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

        self.mlp.train()
        self.gnn.train()

        # hs = torch.ones().to(self.device)
        accs = 0
        loss = 0
        loss_pol = 0
        bn = 0 # batch number

        # for batch in train_loader:
        for i, data in enumerate(train_loader, 0):
            # get batch
            inputs, labels = data
            inputs = inputs.view(-1, self.num_input).to(device)
            adj = torch.Tensor(self.adj)
            if bn == 0:
                # run the first mlp
                y_pred, hs = self.mlp(inputs)
                hs = torch.cat(tuple(hs[i] for i in range(len(hs))), dim=1) # changing dimension to 1 for putting hs vector in gnn
                us = self.gnn(hs, adj) # run the gnn
                print(1)
            else:
                # run the second mlp
                y_pred, hs = self.mlp(inputs, cond_drop=True, us=us)
                hs = torch.cat(tuple(hs[i] for i in range(len(hs))), dim=1)  # changing dimension to 1 for putting hs vector in gnn
                us = self.gnn(hs, adj) # run the gnn
                print(1)
            bn += 1

            # cal acc and rewards for pol
            # calculate accuracy
            y_pred = torch.argmax(y_pred.to('cpu'), dim=1) # why should it be used??
            acc = torch.sum(y_pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # compute mlp loss, loss_pol(policy gradient maybe?)
            # [mlp loss]
            c = nn.CrossEntropyLoss()(y_pred.float(), labels.to(device).float())

            # [loss_pol]
            # Compute the regularization loss L
            # policies => p_i => hidden activity / layer_masks => u_i => Bernoulli(p_i)
            # Todo : Loss 계산 방식 (행 평균, 열 평균)
            L = c + self.lambda_s * (
                    torch.pow(hs.mean(axis=0) - torch.tensor(self.tau).to(device), 2).mean() +
                    torch.pow(hs.mean(axis=1) - torch.tensor(self.tau).to(device), 2).mean())
            L += self.lambda_v * (-1) * (hs.to('cpu').var(axis=0).mean() +
                                         hs.to('cpu').var(axis=1).mean())
            # Compute loss_pol
            logp = torch.log(hs).sum(axis=1).mean()
            pg = self.lambda_pg * c * (-logp) + L

            # update policy
            self.update_pol(loss=c, loss_pol=pg)

            # addup accuracy and loss
            accs += acc
            loss += c.to('cpu').item()
            loss_pol += pg.to('cpu').item()

            # wandb log training/batch
            # wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': pg.item(),
            #            'train/batch_tau': self.tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}, Tau: {:.3f}'
                  .format(1, i, c.item(), pg.item(), acc, us.mean().item()))

        return accs

def main(args):

    # create model
    model = condg_exp(args)
    model.run_exp()




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

    # get time in string to save as file name
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # wandb.init(project="condnet",
    #            config=args.parse_args().__dict__
    #            )
    #
    # wandb.run.name = "condnet_mlp_mnist_{}".format(dt_string)

    main(args=args.parse_args())

    # wandb.finish()