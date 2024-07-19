import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
from torch_geometric.nn import DenseSAGEConv
from tqdm import tqdm
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

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
        layer_cumsum = [0]
        for layer in self.layers:
            layer_cumsum.append(layer.in_features)
        layer_cumsum = np.cumsum(layer_cumsum)

        idx = 0
        if not cond_drop:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                # dropout
                # x = nn.Dropout(p=0.3)(x)
                if i != len(self.layers) - 1:
                    x = F.relu(x)
                    hs.append(x)
        else:
            if us is None:
                raise ValueError('us should be given')
            # conditional activation
            for i, layer in enumerate(self.layers):
                us = us.squeeze()
                if i == 0:
                    # 첫 번째 레이어의 경우 마스킹을 적용하지 않음
                    x = layer(x)
                else:
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    x = layer(x)
                    idx += 1
                if i != len(self.layers) - 1:
                    x = F.relu(x)
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
        # self.mlp_surrogate.load_state_dict(self.mlp.state_dict())

        self.C = nn.CrossEntropyLoss()
    #
    # def forward(self, x):
    #     # x : input
    #     # get policy
    #     u, p = self.gnn(x, adj_)
    #     # get output
    #     y, hs = self.mlp(x, cond_drop=self.compact, us=u)
    #     return y, p, hs, u


def adj(model, bidirect = True, last_layer = False, edge2itself = True):
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
def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=0.005)
    args.add_argument('--lambda_v', type=float, default=0.1)
    args.add_argument('--lambda_l2', type=float, default=1e-5)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=30)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=500)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args = args.parse_args()

    lambda_s = args.lambda_s
    lambda_v = args.lambda_v
    lambda_l2 = args.lambda_l2
    lambda_pg = args.lambda_pg
    tau = args.tau
    learning_rate = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE
    condnet_min_prob = args.condnet_min_prob
    condnet_max_prob = args.condnet_max_prob
    compact = args.compact
    num_inputs = 28**2

    mlp_model = Mlp().to(device)
    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, hidden_size=args.hidden_size).to(device)

    # model = Condnet_model(args=args.parse_args())

    mlp_surrogate = Mlp().to(device)
    # copy weights in mlp to mlp_surrogate
    mlp_surrogate.load_state_dict(mlp_model.state_dict())

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
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    wandb.init(project="condgnet_edit",
                config=args.__dict__,
                name='condg_mlp_mnist_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau)
                )

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(gnn_policy.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=lambda_l2)
    adj_, nodes_ = adj(mlp_model)

    layer_cumsum = [0]
    for layer in mlp_model.layers:
        layer_cumsum.append(layer.in_features)
    layer_cumsum = np.cumsum(layer_cumsum)

    mlp_model.train()
    # run for 50 epochs
    for epoch in trange(max_epochs):
        bn =0
        costs = 0
        accs = 0
        accsbf = 0
        PGs = 0
        num_iteration = 0
        taus = 0
        Ls = 0
        us = torch.zeros((1562, 1562))

        gnn_policy.train()
        mlp_model.train()

        # run for each batch
        for i, data in enumerate(tqdm(train_loader, 0)):

            if args.compact:
                if i>50:
                    break

            mlp_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data
            # get batch

            inputs = inputs.view(-1, num_inputs).to(device)

            # Forward Propagation
            # ouputs, hs     = self.infer_forward_propagation(inputs, adj_)
            # y_pred, us, hs, p = self.forward_propagation(inputs, adj_, hs.detach())
            mlp_surrogate.eval()
            outputs_1, hs = mlp_surrogate(inputs)
            hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
                           dim=1)  # changing dimension to 1 for putting hs vector in gnn
            hs = hs.detach()


            us, p = gnn_policy(hs, adj_)  # run gnn
            outputs, hs = mlp_model(inputs, cond_drop=True, us=us.detach())

            policy_flat = p[:,layer_cumsum[1]:].squeeze()

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels.to(device))
            # Compute the regularization loss L

            Lb_ = torch.pow(policy_flat.mean(axis=0)-torch.tensor(tau).to(device),2).sqrt().sum()
            Le_ = torch.pow(policy_flat.mean(axis=1)-torch.tensor(tau).to(device),2).sqrt().mean()

            Lv_ = -torch.pow(policy_flat - policy_flat.mean(axis=0),2).mean(axis=0).sum()

            L = c + lambda_s * (Lb_ +Le_)

            L += lambda_v * (-1) * (Lv_)

            # Compute the policy gradient (PG) loss
            logp = torch.log(policy_flat).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            PG.backward() # it needs to be checked [TODO]
            mlp_optimizer.step()
            policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            pred_1 = torch.argmax(outputs_1.to('cpu'), dim=1)
            accbf = torch.sum(pred_1 == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            accsbf += accbf
            PGs += PG.to('cpu').item()
            Ls += L.to('cpu').item()

            # surrogate
            mlp_surrogate.load_state_dict(mlp_model.state_dict())

            tau_ = us.mean().detach().item()
            taus += tau_
            # wandb log training/batch
            wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_acc_bf': accbf, 'train/batch_pg': PG.item(), 'train/batch_loss': L.item(), 'train/batch_tau': tau_})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, accbf,tau_ ))

        # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_acc_bf': accsbf / bn, 'train/epoch_tau': taus / bn, 'train/epoch_PG': PGs/bn, 'train/epoch_PG': Ls/bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0
        PGs = 0

        gnn_policy.eval()
        mlp_model.eval()
        with torch.no_grad():

            bn = 0
            costs = 0
            accs = 0
            accsbf = 0
            PGs = 0
            num_iteration = 0
            taus = 0
            Ls = 0
            us = torch.zeros((1562, 1562))

            gnn_policy.train()
            mlp_model.train()

            # run for each batch
            for i, data in enumerate(test_loader, 0):

                bn += 1
                # get batch
                inputs, labels = data
                # get batch

                inputs = inputs.view(-1, num_inputs).to(device)

                # Forward Propagation
                # ouputs, hs     = self.infer_forward_propagation(inputs, adj_)
                # y_pred, us, hs, p = self.forward_propagation(inputs, adj_, hs.detach())
                mlp_surrogate.eval()
                outputs_1, hs = mlp_surrogate(inputs)
                hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
                               dim=1)  # changing dimension to 1 for putting hs vector in gnn
                hs = hs.detach()

                us, p = gnn_policy(hs, adj_)  # run gnn
                outputs, hs = mlp_model(inputs, cond_drop=True, us=us.detach())

                policy_flat = p[:, layer_cumsum[1]:].squeeze()

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                c = C(outputs, labels.to(device))
                # Compute the regularization loss L

                Lb_ = torch.pow(policy_flat.mean(axis=0) - torch.tensor(tau).to(device), 2).sqrt().sum()
                Le_ = torch.pow(policy_flat.mean(axis=1) - torch.tensor(tau).to(device), 2).sqrt().mean()

                Lv_ = -torch.pow(policy_flat - policy_flat.mean(axis=0), 2).mean(axis=0).sum()

                L = c + lambda_s * (Lb_ + Le_)

                L += lambda_v * (-1) * (Lv_)

                # Compute the policy gradient (PG) loss
                logp = torch.log(p.squeeze()).sum(axis=1).mean()
                PG = lambda_pg * c * (-logp) + L

                # calculate accuracy
                pred = torch.argmax(outputs.to('cpu'), dim=1)
                acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
                pred_1 = torch.argmax(outputs_1.to('cpu'), dim=1)
                accbf = torch.sum(pred_1 == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # addup loss and acc
                costs += c.to('cpu').item()
                accs += acc
                accsbf += accbf
                PGs += PG.to('cpu').item()
                Ls += L.to('cpu').item()

                tau_ = us.mean().detach().item()
                taus += tau_

            # wandb log training/epoch
            wandb.log({'test/epoch_cost': costs / bn, 'test/epoch_acc': accs / bn, 'test/epoch_acc_bf': accsbf / bn,
                       'test/epoch_tau': taus / bn, 'test/epoch_PG': PGs / bn, 'test/epoch_L': Ls / bn})
        # save model
        torch.save(mlp_model.state_dict(), './condg_mlp_model_' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
        torch.save(gnn_policy.state_dict(), './condg_gnn_policy_' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')

    wandb.finish()

if __name__=='__main__':
    main()