import torch
import os
import numpy as np
import torch.optim as optim
# import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
from torch_geometric.nn import DenseSAGEConv
from tqdm import tqdm
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

from datetime import datetime

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

sampling_size = (7, 7)

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet_model = models.resnet18(weights=None)
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
        hs = [torch.flatten(F.interpolate(x, size=sampling_size), 2).to(device)]

        layer_cumsum = [0]
        conv_layers = []
        conv_layers.append(self.conv1)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for bottleneck in layer:
                conv_layers.append(bottleneck.conv1)
                conv_layers.append(bottleneck.conv2)
                # if bottleneck.downsample is not None:
                #     for downsample_layer in bottleneck.downsample:
                #         if isinstance(downsample_layer, nn.Conv2d):
                #             conv_layers.append(downsample_layer)

        for layer in conv_layers:
            layer_cumsum.append(layer.in_channels)
        layer_cumsum.append(conv_layers[-1].out_channels)
        layer_cumsum.append(self.fc.out_features)
        layer_cumsum = np.cumsum(layer_cumsum)
        idx = 0
        if not cond_drop:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            hs.append(torch.flatten(F.interpolate(x, size=sampling_size), 2).to(device))
            x = self.max_pool(x)
            count = 0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=sampling_size), 2).to(device))
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    hs.append(torch.flatten(F.interpolate(out, size=sampling_size), 2).to(device))
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                        # hs.append(torch.flatten(F.interpolate(residual, size=sampling_size), 2).to(device))
                    out += residual
                    out = bottleneck.relu(out)
                    # hs.append(torch.flatten(F.interpolate(out, size=sampling_size), 2).to(device))
                    x = out
                count += 1
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            hs.append(x)
        else:
            if us is None:
                raise ValueError('u should be given')
            us = us.squeeze()
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1).to(device)
            x = x * us_
            hs.append(x)
            x = self.max_pool(x)
            idx += 1
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out)
                    us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1).to(device)
                    out = out * us_
                    idx += 1
                    hs.append(out)
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1).to(device)
                    out = out * us_
                    idx += 1
                    hs.append(out)
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                        # us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                        # residual = residual * us_
                    out += residual
                    out = bottleneck.relu(out)
                    # us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    # out = out * us_
                    # i += 1
                    # hs.append(out)
                    x = out

            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].to(device)
            x = x * us_

        return x, hs

class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, batch, conv_len, fc_len, adj_nodes, hidden_size = 64, device ='cpu'):
        super().__init__()
        self.conv1 = DenseSAGEConv(7*7, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.minprob = minprob
        self.maxprob = maxprob
        self.conv_len = conv_len
        self.fc_len = fc_len

    def forward(self, hs, batch_adj):
        conv_hs = torch.cat(tuple(hs[i] for i in range(self.conv_len + 1)), dim=1)
        fc_hs = torch.cat(tuple(hs[i] for i in range(self.conv_len + 1, self.conv_len + self.fc_len + 1)), dim=1)
        fc_hs = fc_hs.unsqueeze(-1)
        fc_hs = fc_hs.expand(-1, -1, 49)

        hs_0 = torch.cat([conv_hs, fc_hs], dim=1)

        hs = F.sigmoid(self.conv1(hs_0, batch_adj))
        hs = F.sigmoid(self.conv2(hs, batch_adj))
        hs = F.sigmoid(self.conv3(hs, batch_adj))
        hs_conv = hs
        hs = self.fc1(hs)
        # hs = self.fc2(hs)
        p = F.sigmoid(hs)
        # bernoulli sampling
        p = p * (self.maxprob - self.minprob) + self.minprob
        u = torch.bernoulli(p).to(
            device)  # [TODO] Error : probability -> not a number(nan), p is not in range of 0 to 1

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

def adj(resnet_model, bidirect=True, last_layer=True, edge2itself=True):
    if last_layer:
        conv_layers = []
        fc_layers = []
        conv_layers.append(list(resnet_model.children())[0])
        for layer in list(resnet_model.children())[4:8]:
            for bottleneck in layer:
                try:
                    conv_layers.append(bottleneck.conv1)
                    conv_layers.append(bottleneck.conv2)
                    # if bottleneck.downsample is not None:
                    #     for downsample_layer in bottleneck.downsample:
                    #         if isinstance(downsample_layer, nn.Conv2d):
                    #             conv_layers.append(downsample_layer)
                except Exception:
                    continue
        fc_layers.append(list(resnet_model.children())[-1])
        num_nodes = (sum([layer.in_channels for layer in conv_layers]) + conv_layers[-1].out_channels) + sum([layer.out_features for layer in fc_layers])
        nl = len(conv_layers) + len(fc_layers)
        conv_trainable_nodes = np.concatenate((np.ones(sum([layer.in_channels for layer in conv_layers])), np.zeros(conv_layers[-1].out_channels)), axis=0)
        fc_trainable_nodes = np.ones(sum([layer.out_features for layer in fc_layers]))
        trainable_nodes = np.concatenate((conv_trainable_nodes, fc_trainable_nodes), axis=0)
        conv_len = len(conv_layers)
        fc_len = len(fc_layers)

    else:
        conv_layers = []
        fc_layers = []
        conv_layers.append(list(resnet_model.children())[0])
        for layer in list(resnet_model.children())[4:8]:
            for bottleneck in layer:
                try:
                    conv_layers.append(bottleneck.conv1)
                    conv_layers.append(bottleneck.conv2)
                    # if bottleneck.downsample is not None:
                    #     for downsample_layer in bottleneck.downsample:
                    #         if isinstance(downsample_layer, nn.Conv2d):
                    #             conv_layers.append(downsample_layer)
                except Exception:
                    continue
        num_nodes = (sum([layer.in_channels for layer in conv_layers]) + conv_layers[-1].out_channels)
        nl = len(conv_layers) + len(fc_layers)
        conv_trainable_nodes = np.concatenate(
            (np.ones(sum([layer.in_channels for layer in conv_layers])), np.zeros(conv_layers[-1].out_channels)),
            axis=0)
        fc_trainable_nodes = np.ones(sum([layer.out_features for layer in fc_layers]))
        trainable_nodes = np.concatenate((conv_trainable_nodes, fc_trainable_nodes), axis=0)
        conv_len = len(conv_layers)
        fc_len = len(fc_layers)

    adjmatrix = np.zeros((num_nodes, num_nodes), dtype=np.int16)
    current_node = 0
    prev_conv2_out_channels = 0

    for i in range(nl):
        if i < len(conv_layers):  # conv_layer
            layer = conv_layers[i]
            num_current = layer.in_channels
            num_next = layer.out_channels
        elif i == len(conv_layers):  # Conv 레이어의 마지막 출력이 FC 레이어로 연결될 때
            num_current = conv_layers[i - 1].out_channels
            num_next = fc_layers[0].out_features

        for j in range(current_node, current_node + num_current):
            for k in range(current_node + num_current, current_node + num_current + num_next):
                adjmatrix[j, k] = 1

        # 잔차 연결 처리 (이전 블록의 `conv2` 출력과 현재 블록의 `conv` 출력 연결)
        if i % 2 == 0 and i > 0:  # `conv`에 해당하는 레이어에서만 잔차 연결 추가
            residual_idx = prev_conv2_out_channels  # 이전 블록의 `conv` 출력 인덱스
            current_out = (current_node + num_current, current_node + num_current + num_next)  # 현재 블록의 `conv` 출력 인덱스
            adjmatrix[residual_idx, current_out] = 1  # 잔차 연결은 단일 연결로 추가

        if i % 2 == 0:  # 짝수
            prev_conv2_out_channels = (current_node + num_current, current_node + num_current + num_next)  # 현재 `conv` 출력 저장

        # print start and end for j
        print(current_node, current_node + num_current)
        # print start and end for k
        print(current_node + num_current, current_node + num_current + num_next)
        print()
        current_node += num_current

        '''
        # `conv2` 출력 채널 수 업데이트
        if i % 2 == 0:  # `conv2`일 때만 업데이트
            prev_conv2_out_channels = num_next  # 현재 블록의 `conv2` 출력 채널 수를 저장
        '''

    if bidirect:
        adjmatrix += adjmatrix.T

    if edge2itself:
        adjmatrix += np.eye(num_nodes, dtype=np.int16)
    adjmatrix[adjmatrix != 0] = 1
    return adjmatrix, trainable_nodes, conv_len, fc_len
def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=10)
    args.add_argument('--lambda_v', type=float, default=0.3)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--max_epochs', type=int, default=100)
    args.add_argument('--tau', type=float, default=0.3)
    args.add_argument('--condnet_min_prob', type=float, default=0.01)
    args.add_argument('--condnet_max_prob', type=float, default=0.99)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=5) # [TODO]: gradient accumulate step
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=64)
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

    mlp_model = ResNet50().to(device)
    adj_, nodes_, conv_len, fc_len = adj(mlp_model)
    adj_ = torch.stack([torch.Tensor(adj_) for _ in range(BATCH_SIZE)]).to(device)

    gnn_policy = Gnn(args.condnet_min_prob, args.condnet_max_prob, batch=args.BATCH_SIZE,
                     conv_len=conv_len, fc_len=fc_len, adj_nodes=len(nodes_),
                     hidden_size=args.hidden_size).to(device)

    # model = Condnet_model(args=args.parse_args())

    mlp_surrogate = ResNet50().to(device)
    # copy weights in mlp to mlp_surrogate
    mlp_surrogate.load_state_dict(mlp_model.state_dict())

    dataset_path = r'C:\Users\97dnd\anaconda3\envs\torch\pr\condnet-g\data'
    dataset = load_from_disk(dataset_path)

    transform = transforms.Compose([
            # 짧은 변을 [256, 480] 사이에서 무작위로 리사이즈하고, 224x224로 크롭
            transforms.RandomResizedCrop(224, scale=(256 / 480, 1.0)),
            # 무작위로 수평 뒤집기 적용
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    class CustomDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image, label = item['image'], item['label']
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = CustomDataset(dataset['validation'], transform=transform)
    test_dataset = CustomDataset(dataset['validation'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # SimpleCNN 모델 초기화
    # mlp_model = SimpleCNN().to(device)
    #
    # # adj_ 및 nodes_ 계산
    # adj_, nodes_ = adj(mlp_model)
    #
    # # AdjBatchLoader 정의
    # class AdjBatchLoader:
    #     def __init__(self, adj, batch_size, total_batches):
    #         self.adj = adj
    #         self.batch_size = batch_size
    #         self.total_batches = total_batches
    #
    #     def __getitem__(self, index):
    #         if index < self.total_batches - 1:
    #             return torch.stack([torch.Tensor(self.adj) for _ in range(self.batch_size)]).to(device)
    #         else:  # 마지막 배치일 경우
    #             remaining_size = len(self.adj) % self.batch_size
    #             return torch.stack([torch.Tensor(self.adj) for _ in range(remaining_size)]).to(device)
    #
    #     def __len__(self):
    #         return self.total_batches
    #
    # # AdjBatchLoader 초기화
    # total_batches = len(train_loader)
    # adj_batch_loader = AdjBatchLoader(adj_, BATCH_SIZE, total_batches)
    #
    # # GNN 모델 초기화
    # gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, batch=BATCH_SIZE,
    #                  conv_len=len(mlp_model.conv_layers), fc_len=len(mlp_model.fc_layers),
    #                  adj_nodes=len(nodes_), hidden_size=args.hidden_size).to(device)
    #
    # # mlp_surrogate 초기화 및 가중치 복사
    # mlp_surrogate = SimpleCNN().to(device)
    # mlp_surrogate.load_state_dict(mlp_model.state_dict())

    wandb.init(project="condgtest",
                config=args.__dict__,
                name='condg_cnn_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau)
                )

    C = nn.CrossEntropyLoss()
    # mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate,
    #                           momentum=0.9, weight_decay=lambda_l2)
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.0003, weight_decay=1e-4)
    policy_optimizer = optim.SGD(gnn_policy.parameters(), lr=learning_rate,
                                 momentum=0.9, weight_decay=lambda_l2)
    # policy_optimizer = optim.Adam(gnn_policy.parameters(), lr=0.0003, weight_decay=1e-4)

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

            # inputs = inputs.view(-1, num_inputs).to(device)
            inputs = inputs.to(device)

            # adj_batch = adj_batch_loader[i]

            # if inputs.size(0) < BATCH_SIZE:
            #     adj_batch = torch.stack([torch.Tensor(adj_) for _ in range(inputs.size(0))]).to(device)
            # else:
            #     adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

            # Forward Propagation
            # ouputs, hs     = self.infer_forward_propagation(inputs, adj_)
            # y_pred, us, hs, p = self.forward_propagation(inputs, adj_, hs.detach())
            mlp_surrogate.eval()
            outputs_1, hs = mlp_surrogate(inputs)
            current_batch_size = hs[0].shape[0]

            if current_batch_size < BATCH_SIZE:
                adj_batch = adj_[:current_batch_size]
            else:
                adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용
            # print(adj_.shape)
            # print(adj_batch.size())  # 크기 확인
            # adj_ = torch.stack([torch.Tensor(adj_) for _ in range(hs[0].shape[0])]).to(device)

            # hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
            #                dim=1)  # changing dimension to 1 for putting hs vector in gnn
            # hs = hs.detach()


            us, p = gnn_policy(hs, adj_batch)  # run gnn
            outputs, hs = mlp_model(inputs, cond_drop=True, us=us.detach())

            c = C(outputs, labels.to(device))
            # Compute the regularization loss L

            # Lb_ = torch.norm(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), p=2)
            Lb_ = torch.pow(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean()
            Le_ = torch.pow(p.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean()

            L = c + lambda_s * (Lb_)

            Lv_ = -torch.norm(p.squeeze() - p.squeeze().mean(axis=0), p=2, dim=0).mean()
            # Lv_ = (-1)* (p.squeeze().var(axis=0).mean()).mean()

            L += lambda_v * Lv_

            # Compute the policy gradient (PG) loss
            logp = torch.log(p.squeeze()).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L
            # PG = lambda_pg * c * (-logp) + L

            PG.backward()  # it needs to be checked [TODO]
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
            wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_acc_bf': accbf,
                       'train/batch_pg': PG.item(), 'train/batch_loss': L.item(), 'train/batch_tau': tau_,
                       'train/batch_Lb': Lb_, 'train/batch_Le': Le_, 'train/batch_Lv': Lv_})

            # print PG.item(), and acc with name
            print(
                'Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Lb {:.3f}, Lv {:.3f}, Acc: {:.3f}, Acc: {:.3f}, Tau: {:.3f}'.format(
                    epoch, i, c.item(), PG.item(), Lb_, Lv_, acc, accbf, tau_))

            # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_acc_bf': accsbf / bn,
                   'train/epoch_tau': taus / bn, 'train/epoch_PG': PGs / bn, 'train/epoch_PG': Ls / bn})

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
            for i, data in enumerate(tqdm(test_loader, 0)):
                bn += 1
                # get batch
                inputs, labels = data
                # get batch

                inputs = inputs.to(device)

                # adj_batch = adj_batch_loader[i]

                # if inputs.size(0) < BATCH_SIZE:
                #     adj_batch = torch.stack([torch.Tensor(adj_) for _ in range(inputs.size(0))]).to(device)
                # else:
                #     adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

                # Forward Propagation
                # ouputs, hs     = self.infer_forward_propagation(inputs, adj_)
                # y_pred, us, hs, p = self.forward_propagation(inputs, adj_, hs.detach())
                mlp_surrogate.eval()
                outputs_1, hs = mlp_surrogate(inputs)
                # adj_ = torch.stack([torch.Tensor(adj_) for _ in range(hs[0].shape[0])]).to(device)
                # hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
                #                dim=1)  # changing dimension to 1 for putting hs vector in gnn
                # hs = hs.detach()

                current_batch_size = hs[0].shape[0]

                if current_batch_size < BATCH_SIZE:
                    adj_batch = adj_[:current_batch_size]
                else:
                    adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

                # print(adj_batch.size())  # 크기 확인

                us, p = gnn_policy(hs, adj_batch)  # run gnn
                outputs, hs = mlp_model(inputs, cond_drop=True, us=us.detach())

                c = C(outputs, labels.to(device))

                Lb_ = torch.pow(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean()
                Le_ = torch.pow(p.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean()

                L = c + lambda_s * (Lb_)

                Lv_ =  -torch.norm(p.squeeze() - p.squeeze().mean(axis=0), p=2, dim=0).mean()
                L += lambda_v * Lv_

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
        torch.save(mlp_model.state_dict(),
                   './resnet_model_' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(
                       args.tau) + dt_string + '.pt')
        torch.save(gnn_policy.state_dict(),
                   './gnn_policy_' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(
                       args.tau) + dt_string + '.pt')

    wandb.finish()

if __name__ == '__main__':
    main()