import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import DenseSAGEConv
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import wandb
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# wandb.init(project="condgnet",entity='hails', name='resnet50_imagenet')
# wandb.login(key="651ddb3adb37c78e1ae53ac7709b316915ee6909")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet_model = models.resnet50(pretrained=True)
        self.modules = list(resnet_model.children())
        self.len_modules = len(self.modules)
        self.conv1 = self.modules[0]
        self.bn1 = self.modules[1]
        self.relu = self.modules[2]
        self.max_pool = self.modules[3]
        self.layer1 = self.modules[4]
        self.layer2 = self.modules[5]
        self.layer3 = self.modules[6]
        self.layer4 = self.modules[7]
        self.avg_pool = self.modules[8]
        self.fc = self.modules[9]

    def forward(self, x, cond_drop=False, us=None, channels=None):
        hs = [torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(device)]
        if not cond_drop:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            hs.append(torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(device))
            x = self.max_pool(x)
            # for layer in [self.conv1, self.bn1, self.relu, self.max_pool]:
            #     x = layer(x)
            #     if 'ReLU' in str(layer):
            #         hs.append(torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(device))
            count=0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=(7, 7)), 2).to(device))
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=(7, 7)), 2).to(device))
                    out = bottleneck.conv3(out)
                    out = bottleneck.bn3(out)
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                    out += residual
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=(7, 7)), 2).to(device))
                    x = out
                count+=1
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            if us is None:
                raise ValueError('u should be given')
            us = us.unsqueeze(-1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            us_ = us[:, channels[0]:channels[0] + channels[1]]
            x = x * us_
            hs.append(x)
            x = self.max_pool(x)
            # for layer in [self.conv1, self.bn1, self.relu, self.max_pool]:
                # x = layer(x)
                # if 'Conv' in str():
                #     x = x * us[:, channels[0]: channels[0] + channels[1]]
                # elif 'ReLU' in str(layer):
                #     hs.append(torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(device))

            i = 0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    # i = 0
                    # l_idx = []
                    # i_idx = []
                    # for l in bottleneck.children():
                    #     l_idx.append(l)
                    #     i_idx.append(i)
                    #     print(channels[i + 2])
                    #     i+=1


                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out) # 64

                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1

                    hs.append(out)
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    out = bottleneck.relu(out) # 64

                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1

                    hs.append(out)
                    out = bottleneck.conv3(out)
                    out = bottleneck.bn3(out)
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                    out += residual
                    out = bottleneck.relu(out) # 256

                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1

                    hs.append(out)
                    x = out

                    # i = 0
                    # for l in bottleneck.children():
                    #     if 'Conv' in str(l):
                    #         out = out * us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    #         i += 1

            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x, hs

def adj(resnet_model, bidirect = True, last_layer = True, edge2itself = True):
    # resnet_model = models.resnet50(pretrained=True)
    if last_layer:
        num_channels_ls = []
        for i in range(len(list(resnet_model.children())[0:4])):
            try:
                num_channels_ls.append(list(resnet_model.children())[0:4][i].in_channels)
            except Exception:
                continue
        for layer in list(resnet_model.children())[4:8]:
            for bottleneck in layer:
                try:
                    num_channels_ls.append(bottleneck.conv1.in_channels)
                    num_channels_ls.append(bottleneck.conv2.in_channels)
                    num_channels_ls.append(bottleneck.conv3.in_channels)
                except Exception:
                    continue
                    # for l in bottleneck.children():
                    #     if isinstance(l, torch.nn.modules.conv.Conv2d):
                    #         num_channels_ls.append(l.in_channels)
                    #     elif isinstance(l, torch.nn.Sequential):
                    #         for sub_layer in l.children():
                    #             if isinstance(sub_layer, torch.nn.modules.conv.Conv2d):
                    #                 num_channels_ls.append(sub_layer.in_channels)
                #     for l in bottleneck.children():
                #         if 'Conv' in str(l):
                #             num_channels_ls.append(l.in_channels)
                # except Exception:
                #     continue
        num_channels_ls.append(list(resnet_model.children())[7][2].conv3.out_channels)
        num_channels = sum(num_channels_ls)
        num_conv_layers = len(num_channels_ls)

    else:
        num_channels_ls = []
        for i in range(len(list(resnet_model.children())[0:4])):
            try:
                num_channels_ls.append(list(resnet_model.children())[0:4][i].in_channels)
            except Exception:
                continue
        for layer in list(resnet_model.children())[4:8]:
            for bottleneck in layer:
                try:
                    num_channels_ls.append(bottleneck.conv1.in_channels)
                    num_channels_ls.append(bottleneck.conv2.in_channels)
                    num_channels_ls.append(bottleneck.conv3.in_channels)
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

        if i % 3 == 0 and i != 0:
            for j in range(current_node, current_node + num_current):
                for k in range(current_node - num_channels_ls[i - 3], current_node + num_current):
                    adjmatrix[j, k] = 1

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
        self.conv1 = DenseSAGEConv(7*7, hidden_size)
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
    # args.add_argument('--condnet_min_prob', type=float, default=0.01)
    # args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--learning_rate', type=float, default=0.001)
    args.add_argument('--BATCH_SIZE', type=int, default=8)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=256)
    args.add_argument('--accum-step', type=int, default=8)
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

    # trainset = torchvision.datasets.ImageNet('./data', split='train', transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    #
    # testset = torchvision.datasets.ImageNet('./data', split='val', transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset = ImageFolder('C:/Users/97dnd/anaconda3/envs/torch/pr/resnet/data/ILSVRC2012_img_train',
                                transform=transform)
    val_dataset = ImageFolder('C:/Users/97dnd/anaconda3/envs/torch/pr/resnet/data/ILSVRC2012_img_val',
                              transform=transform)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    resnet = ResNet50()
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = True

    resnet_model = ResNet50()
    resnet_model = resnet_model.to(device)
    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, hidden_size=args.hidden_size).to(device)
    adj_, num_channels_ls = adj(resnet_model)

    criterion = nn.CrossEntropyLoss()
    resnet_optimizer = optim.SGD(resnet_model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(gnn_policy.parameters(), lr=learning_rate,
                                 momentum=0.9, weight_decay=lambda_l2)

    num_params = 0
    for param in gnn_policy.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in resnet_model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    resnet_surrogate = ResNet50().to(device)
    # copy weights in mlp to mlp_surrogate
    resnet_surrogate.load_state_dict(resnet_model.state_dict())

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
        resnet_model.train()

        resnet_optimizer.zero_grad()
        policy_optimizer.zero_grad()

        L_accum = []
        c_accum = []
        PG_accum = []
        acc_accum = []
        accbf_accum = []
        tau_accum = []


        for i, data in enumerate(train_loader, start=0):

            if args.compact:
                if i > 50:
                    break

            bn += 1

            resnet_surrogate.eval()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs_surrogate, hs = resnet_surrogate(inputs)

            us, p = gnn_policy(hs, adj_)  # run gnn
            outputs, hs = resnet_model(inputs, cond_drop=True, us=us.detach(), channels=num_channels_ls)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 1000)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1
            # y_one_hot = torch.zeros((2, 1000), device=device)
            # y_one_hot[torch.arange(labels.shape[0], device=device), labels] = 1
            c = criterion( F.softmax(outputs, dim=1), labels.to(device))
            # Compute the regularization loss L

            L = c + lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean() +
                                torch.pow(p.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean())

            L += lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                    p.squeeze().var(axis=1).mean())

            # Compute the policy gradient (PG) loss
            logp = torch.log(p.squeeze()).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            pred = pred.to(device)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            pred_1 = torch.argmax(outputs_surrogate, dim=1)
            accbf = torch.sum(pred_1 == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            tau_ = us.mean().detach().item()
            L_accum.append(L.item())
            c_accum.append(c.item())
            PG_accum.append(PG.item())
            acc_accum.append(acc)
            accbf_accum.append(accbf)
            tau_accum.append(tau_)

            PG /= args.accum_step
            PG.backward()

            if (i+1) % args.accum_step ==0 or ((i+1)==len(train_loader)):
                resnet_optimizer.step()
                policy_optimizer.step()

                # addup loss and acc
                costs += np.sum(c_accum)
                accs += np.mean(acc_accum)
                accsbf += np.mean(accbf_accum)
                PGs += np.sum(PG_accum)
                Ls += np.sum(L_accum)

                # surrogate
                resnet_surrogate.load_state_dict(resnet_model.state_dict())

                taus += np.mean(tau_accum)

                # print PG.item(), and acc with name
                # print(
                    # 'Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Accbf: {:.3f}, Tau: {:.3f}'.format(epoch, i,
                    #                                                                                                c.item(),
                    #                                                                                                PG.item(),
                    #                                                                                                acc,
                    #                                                                                                accbf,
                    #                                                                                                tau_))
                print(
                    'Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Accbf: {:.3f}, Tau: {:.3f}'.format(epoch, i, np.sum(c_accum)/args.accum_step, np.sum(PG_accum)/args.accum_step, np.mean(acc_accum), np.mean(accbf_accum), np.mean(tau_accum)))


                resnet_optimizer.zero_grad()
                policy_optimizer.zero_grad()

                L_accum = []
                c_accum = []
                PG_accum = []
                acc_accum = []
                accbf_accum = []
                tau_accum = []


        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

if __name__=='__main__':
    main()