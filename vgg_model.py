import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import DenseSAGEConv
from torchvision import models
# import cv2
import numpy as np
from datetime import datetime
import json
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
# dataset = load_dataset("imagenet-1k")
# https://huggingface.co/datasets/imagenet-1k

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def downsample(img, size):
#     ds = []
#     for i in range(img.shape[0]):
#         # shape : (b, c, h, w) -> (c, h, w)
#         # input : (h, w, c)
#         # resized : (h, w, c)
#         ds.append(cv2.resize(img[i].cpu().numpy().transpose(1, 2, 0), dsize=size, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))
#     return torch.tensor(np.array(ds))
class Vgg(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.modules = list(vgg_model.children())
        self.conv = self.modules[0]
        self.avg_pool = self.modules[1]
        self.classifier = self.modules[2]

    def forward(self, x, cond_drop=False, us=None, channels=None):
        hs = [torch.flatten(F.interpolate(x, size=(56,56)),2).to(device)] # [TODO]: Down/Up sampling size

        # hs = []
        if not cond_drop:
            for i, layer in enumerate(self.conv):
                x = layer(x)
                if str(layer) == 'ReLU(inplace=True)':
                    hs.append(torch.flatten(F.interpolate(x, size=(56,56)),2).to(device))
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            if us is None:
                raise ValueError('u should be given')
            # conditional activation
            us = us.unsqueeze(-1)
            # x = x * us[:, 0:channels[0]]  # torch.Size([batch, 4227, 1]) # [TODO]: input image also need channel wise pruning ?
            i = 0
            for layer in self.conv:
                x = layer(x)
                if 'Conv' in str(layer):
                    x = x * us[:,channels[i]:channels[i]+channels[i+1]]
                    i += 1
                elif str(layer) == 'ReLU(inplace=True)':
                    hs.append(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x, hs

def adj(vgg_model, bidirect = True, last_layer = True, edge2itself = True):
    # if last_layer:
    #     num_nodes = sum([layer.in_features for layer in model.layers]) + model.layers[-1].out_features
    #     nl = len(model.layers)
    #     trainable_nodes = np.concatenate(
    #         (np.ones(sum([layer.in_features for layer in model.layers])), np.zeros(model.layers[-1].out_features)),
    #         axis=0)
    #     # trainable_nodes => [1,1,1,......,1,0,0,0] => input layer & hidden layer 의 노드 개수 = 1의 개수, output layer 의 노드 개수 = 0의 개수
    # else:
    #     num_nodes = sum([layer.in_features for layer in model.layers])
    #     nl = len(model.layers) - 1
    #     trainable_nodes = np.ones(num_nodes)

    if last_layer:
        num_channels_ls = []
        for i in range(len(list(vgg_model.children())[0])):
            try:
                num_channels_ls.append(list(vgg_model.children())[0][i].in_channels)
            except Exception:
                continue
        num_channels_ls.append(list(vgg_model.children())[0][-3].out_channels)
        num_channels = sum(num_channels_ls)
        num_conv_layers = len(num_channels_ls)
    else:
        num_channels_ls = []
        for i in range(len(list(vgg_model.children())[0])):
            try:
                num_channels_ls.append(list(vgg_model.children())[0][i].in_channels)
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

class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, hidden_size = 64):
        super().__init__()
        self.conv1 = DenseSAGEConv(56*56, hidden_size)
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
    args.add_argument('--max_epochs', type=int, default=50)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=64)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=256)
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.ImageNet('./data/imagenet', split='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = torchvision.datasets.ImageNet('./data/imagenet', split='val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    vgg16 = models.vgg16(weights='IMAGENET1K_V1')
    vgg16 = vgg16.to(device)
    for param in vgg16.features.parameters():
        param.requires_grad = True

    vgg_model = Vgg(vgg16)
    vgg_model = vgg_model.to(device)
    print()
    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, hidden_size=args.hidden_size).to(device)
    adj_, num_channels_ls = adj(vgg_model)

    criterion = nn.CrossEntropyLoss()
    vgg_optimizer = optim.SGD(vgg_model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(gnn_policy.parameters(), lr=learning_rate,
                                 momentum=0.9, weight_decay=lambda_l2)

    num_params = 0
    for param in gnn_policy.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in vgg_model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    vgg_surrogate = Vgg(vgg16).to(device)
    # copy weights in mlp to mlp_surrogate
    vgg_surrogate.load_state_dict(vgg_model.state_dict())

    # run for 50 epochs
    for epoch in range(max_epochs):
        bn = 0
        costs = 0
        accs = 0
        accsbf = 0
        PGs = 0
        num_iteration = 0
        taus = 0
        Ls = 0
        gnn_policy.train()
        vgg_model.train()

        for i, data in enumerate(trainloader, start=0):

            if args.compact:
                if i > 50:
                    break

            vgg_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1

            vgg_surrogate.eval()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs_surrogate, hs = vgg_surrogate(inputs)

            us, p = gnn_policy(hs, adj_)  # run gnn
            outputs, hs = vgg_model(inputs, cond_drop=True, us=us.detach(), channels=num_channels_ls)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 1000)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = criterion(outputs, labels.to(device))
            # Compute the regularization loss L

            L = c + lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean() +
                                torch.pow(p.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean())

            L += lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                    p.squeeze().var(axis=1).mean())

            # Compute the policy gradient (PG) loss
            logp = torch.log(p.squeeze()).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            PG.backward()
            vgg_optimizer.step()
            policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            pred = pred.to(device)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            pred_1 = torch.argmax(outputs_surrogate, dim=1)
            accbf = torch.sum(pred_1 == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            accsbf += accbf
            PGs += PG.to('cpu').item()
            Ls += L.to('cpu').item()

            # surrogate
            vgg_surrogate.load_state_dict(vgg_model.state_dict())

            tau_ = us.mean().detach().item()
            taus += tau_

            # print PG.item(), and acc with name
            print(
                'Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Accbf: {:.3f}, Tau: {:.3f}'.format(epoch, i,
                                                                                                               c.item(),
                                                                                                               PG.item(),
                                                                                                               acc,
                                                                                                               accbf,
                                                                                                               tau_))

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

if __name__=='__main__':
    main()