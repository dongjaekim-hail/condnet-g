import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

from torch_geometric.nn import DenseSAGEConv

import torch.nn as nn
import torch.nn.functional as F
import wandb

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
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64,1)
        self.minprob = minprob
        self.maxprob = maxprob

    def forward(self, hs, adj):
        # hs : hidden activity
        batch_adj = torch.stack([torch.Tensor(adj) for _ in range(hs.shape[0])])
        batch_adj = batch_adj.to(device)

        hs = F.tanh(self.conv1(hs.unsqueeze(-1), batch_adj))
        hs = F.tanh(self.conv2(hs, batch_adj))
        hs = F.tanh(self.conv3(hs, batch_adj))
        hs = F.tanh(self.fc1(hs))
        hs = self.fc2(hs)
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
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE
    condnet_min_prob = args.condnet_min_prob
    condnet_max_prob = args.condnet_max_prob
    compact = args.compact
    num_inputs = 28**2

    mlp_model = Mlp().to(device)
    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob).to(device)

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
        us = torch.zeros((1562, 1562))

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
            outputs, hs = mlp_surrogate(inputs)
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
            PG = lambda_pg * c/p.squeeze().mean() * (-logp) + L
            # PG = lambda_pg * c * (-logp) + L

            PG.backward() # it needs to be checked [TODO]
            mlp_optimizer.step()
            policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            PGs += PG.to('cpu').item()

            # surrogate
            mlp_surrogate.load_state_dict(mlp_model.state_dict())

            # wandb log training/batch
            # wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': PG.item(), 'train/batch_tau': tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, us.mean().detach().item()))

        # wandb log training/epoch
        # wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': tau, 'train/epoch_PG': PGs/bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0
        PGs = 0

        # model.eval()
        # with torch.no_grad():
        #     # calculate accuracy on test set
        #     acc = 0
        #     bn = 0
        #     for i, data in enumerate(test_loader, 0):
        #         bn += 1
        #         # get batch
        #         inputs, labels = data
        #
        #         # make one hot vector
        #         y_batch_one_hot = torch.zeros(labels.shape[0], 10)
        #         y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1,).tolist()] = 1
        #
        #         # get output
        #         outputs, policies, sample_probs, layer_masks = model(torch.tensor(inputs))
        #
        #         # calculate accuracy
        #         pred = torch.argmax(outputs, dim=1).to('cpu')
        #         acc  = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
        #
        #         # make labels one hot vector
        #         y_one_hot = torch.zeros(labels.shape[0], 10)
        #         y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1
        #
        #         c = C(outputs, labels.to(model.device))
        #
        #         # Compute the regularization loss L
        #
        #         L = c + lambda_s * (torch.pow(torch.stack(policies).mean(axis=1) - torch.tensor(tau).to(model.device), 2).mean() +
        #                             torch.pow(torch.stack(policies).mean(axis=2) - torch.tensor(tau).to(model.device), 2).mean())
        #
        #         L += lambda_v * (-1) * (torch.stack(policies).var(axis=1).mean() +
        #                                 torch.stack(policies).var(axis=2).mean())
        #
        #
        #
        #         # Compute the policy gradient (PG) loss
        #         logp = torch.log(torch.cat(policies)).sum(axis=1).mean()
        #         PG = lambda_pg * c * (-logp) + L
        #
        #         # wandb log test/batch
        #         wandb.log({'test/batch_acc': acc, 'test/batch_cost': c.to('cpu').item(), 'test/batch_pg': PG.to('cpu').item()})
        #
        #         # addup loss and acc
        #         costs += c.to('cpu').item()
        #         accs += acc
        #         PGs += PG.to('cpu').item()
        #     #print accuracy
        #     print('Test Accuracy: {}'.format(accs / bn))
        #
        #     # wandb log test/epoch
        #     wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn, 'test/epoch_pg': PGs / bn})

if __name__=='__main__':
    # make arguments and defaults for the parameters
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=10)
    args.add_argument('--lambda_v', type=float, default=10)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.2)
    args.add_argument('--max_epochs', type=int, default=1000)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.6)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=256)
    args.add_argument('--compact', type=bool, default=True)

    # get time in string to save as file name
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # wandb.init(project="condnet",
    #             config=args.parse_args().__dict__
    #             )

    # wandb.run.name = "condnet_mlp_mnist_{}".format(dt_string)

    main(args=args.parse_args())

    # wandb.finish()