import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

from torch_geometric.nn import DenseSAGEConv

import torch.nn as nn
import torch.nn.functional as F
# import wandb
import torch

from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

# class Mlp(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(32*32*3, 512))
#         self.layers.append(nn.Linear(512, 256))
#         self.layers.append(nn.Linear(256, 10))
#
#     def forward(self, x, cond_drop=False, us=None):
#         hs = [x]
#         # flatten
#         len_in = 0
#         len_out = x.shape[1]
#         if not cond_drop:
#             for layer in self.layers:
#                 x = layer(x)
#                 x = F.sigmoid(x)
#                 # dropout
#                 # x = nn.Dropout(p=0.3)(x)
#                 hs.append(x)
#         else:
#             if us is None:
#                 raise ValueError('u should be given')
#             # conditional activation
#             for layer in self.layers:
#                 us = us.squeeze()
#                 len_out = layer.in_features
#                 x = x * us[:,len_in:len_in+len_out] # where it cuts off [TODO]
#                 x = layer(x)
#                 x = F.relu(x)
#                 # dropout
#                 # x = nn.Dropout(p=0.3)(x)
#                 len_in = len_out
#                 hs.append(x)
#
#         # softmax
#         x = F.softmax(x, dim=1)
#         return x, hs

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, cond_drop=False, us=None, channels=None):
        hs = [torch.flatten(F.interpolate(x, size=(16, 16)), 2).to(device)]

        if not cond_drop:
            x = self.conv1(x)
            x = F.relu(x)
            hs.append(torch.flatten(F.interpolate(x, size=(16, 16)), 2).to(device))
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            hs.append(torch.flatten(F.interpolate(x, size=(16, 16)), 2).to(device))
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc1(x)
            x = F.relu(x)
            hs.append(torch.flatten(F.interpolate(x, size=(16, 16)), 2).to(device))
            x = self.dropout(x)
            x = self.fc2(x)

        else:
            if us is None:
                raise ValueError('u should be given')
            us = us.unsqueeze(-1)
            x = self.conv1(x)
            x = F.relu(x)
            x = x * us[:, channels[i]:channels[i] + channels[i + 1]]
            hs.append(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = x * us[:, channels[i]:channels[i] + channels[i + 1]]
            hs.append(x)
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc1(x)
            x = F.relu(x)
            x = x * us[:, channels[i]:channels[i] + channels[i + 1]]
            hs.append(x)
            x = self.dropout(x)
            x = self.fc2(x)

        return x, hs


class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, hidden_size = 64):
        super().__init__()
        self.conv1 = DenseSAGEConv(14*14, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(64,1, bias=False)
        self.minprob = minprob
        self.maxprob = maxprob

    def forward(self, hs, adj):
        # hs : hidden activity
        h = hs[0]
        for i in hs[1:]:
            h = torch.cat((h, i), dim=1)
        h = h.to(device)

        batch_adj = torch.stack([torch.Tensor(adj) for _ in range(h.shape[0])])
        batch_adj = batch_adj.to(device)

        # hs_0 = hs.unsqueeze(-1)

        hs = F.sigmoid(self.conv1(h, batch_adj))
        hs = F.sigmoid(self.conv2(hs, batch_adj))
        hs = F.sigmoid(self.conv3(hs, batch_adj))
        hs = self.fc1(hs)
        p = F.sigmoid(hs)
        # bernoulli sampling
        p = p * (self.maxprob - self.minprob) + self.minprob
        u = torch.bernoulli(p).to(device)

        return u, p

def adj(model, bidirect = True, last_layer = True, edge2itself = True):
    if last_layer:
        num_channels_ls = []
        for i in range(len(list(model.children()))):
            try:
                num_channels_ls.append(list(model.children())[i].in_channels)
            except Exception:
                continue
        num_channels_ls.append(model.conv2.out_channels)
        num_channels = sum(num_channels_ls)
        num_conv_layers = len(num_channels_ls)
    else:
        num_channels_ls = []
        for i in range(len(list(model.children()))):
            try:
                num_channels_ls.append(list(model.children())[i].in_channels)
            except Exception:
                continue
        num_channels = sum(num_channels_ls)
        num_conv_layers = len(num_channels_ls)

    adjmatrix = np.zeros((num_channels, num_channels), dtype=np.int16)
    current_node = 0

    for i in range(num_conv_layers - 1):
        # layer = model.layers[i]
        # num_current = layer.in_features
        # num_next = layer.out_features
        num_current = num_channels_ls[i]
        num_next = num_channels_ls[i + 1]

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
        adjmatrix += np.eye(num_channels, dtype=np.int16)
        # make sure every element that is non-zero is 1
    adjmatrix[adjmatrix != 0] = 1
    return adjmatrix, num_channels_ls
def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=7)
    args.add_argument('--lambda_v', type=float, default=1.2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=40)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=256)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args = args.parse_args()

    lambda_s = args.lambda_s
    lambda_v = args.lambda_v
    lambda_l2 = args.lambda_l2
    lambda_pg = args.lambda_pg
    tau = args.tau
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE
    condnet_min_prob = args.condnet_min_prob
    condnet_max_prob = args.condnet_max_prob
    compact = args.compact
    num_inputs = 32*32*3

    mlp_model = Mlp().to(device)
    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, hidden_size=args.hidden_size).to(device)

    # model = Condnet_model(args=args.parse_args())

    num_params = 0
    for param in gnn_policy.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in mlp_model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))


    mlp_surrogate = Mlp().to(device)
    # copy weights in mlp to mlp_surrogate
    mlp_surrogate.load_state_dict(mlp_model.state_dict())

    # datasets load mnist data
    train_dataset = datasets.CIFAR10(
        root="../data/cifar10",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.CIFAR10(
        root="../data/cifar10",
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

    # wandb.init(project="condgnet",
    #             config=args.__dict__,
    #             name='cifar10'
    #             )

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(gnn_policy.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=lambda_l2)
    adj_, nodes_ = adj(mlp_model)

    mlp_model.train()
    # run for 50 epochs
    for epoch in range(max_epochs):
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
        for i, data in enumerate(train_loader, 0):

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


            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels.to(device))
            # Compute the regularization loss L

            L = c + lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean() +
                                torch.pow(p.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean())

            L += lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                    p.squeeze().var(axis=1).mean())



            # Compute the policy gradient (PG) loss
            logp = torch.log(p.squeeze()).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L
            # PG = lambda_pg * c * (-logp) + L

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
            # wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_acc_bf': accbf, 'train/batch_pg': PG.item(), 'train/batch_loss': L.item(), 'train/batch_tau': tau_})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, accbf,tau_ ))

        # wandb log training/epoch
        # wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_acc_bf': accsbf / bn, 'train/epoch_tau': taus / bn, 'train/epoch_PG': PGs/bn, 'train/epoch_PG': Ls/bn})

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

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                c = C(outputs, labels.to(device))
                # Compute the regularization loss L

                L = c + lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean() +
                                    torch.pow(p.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean())

                L += lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                        p.squeeze().var(axis=1).mean())

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
            # wandb.log({'test/epoch_cost': costs / bn, 'test/epoch_acc': accs / bn, 'test/epoch_acc_bf': accsbf / bn,
            #            'test/epoch_tau': taus / bn, 'test/epoch_PG': PGs / bn, 'test/epoch_L': Ls / bn})
        # save model
        torch.save(mlp_model.state_dict(), './10mlp_model_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
        torch.save(gnn_policy.state_dict(), './10gnn_policy_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')

    # wandb.finish()

if __name__=='__main__':
    main()