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

from datetime import datetime

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

sampling_size = (8, 8)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.pooling_layer = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.conv_layers.append(nn.Conv2d(3, 64, 3, padding=1)) # conv1
        self.conv_layers.append(nn.Conv2d(64, 64, 3, padding=1)) # conv2
        self.conv_layers.append(nn.Conv2d(64, 128, 3, padding=1)) # conv3
        self.conv_layers.append(nn.Conv2d(128, 128, 3, padding=1)) # conv4

        self.fc_layers.append(nn.Linear(128 * 8 * 8, 256)) # fc1 # Assuming input image size is 32x32
        self.fc_layers.append(nn.Linear(256, 256)) # fc2
        self.fc_layers.append(nn.Linear(256, 10)) # fc3

        '''
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
        '''

    def forward(self, x, cond_drop=False, us=None):
        hs = [torch.flatten(F.interpolate(x, size=sampling_size), 2).to(device)]

        layer_cumsum = [0]
        for layer in self.conv_layers:
            layer_cumsum.append(layer.in_channels)
        layer_cumsum.append(self.conv_layers[-1].out_channels)
        for layer in self.fc_layers:
            layer_cumsum.append(layer.out_features)
        layer_cumsum = np.cumsum(layer_cumsum)
#array([  0,   3,  67, 131, 259, 387, 643, 899, 909])
        idx = 0
        if not cond_drop:
            for i, layer in enumerate(self.conv_layers, start=1):
                if i == len(self.conv_layers):
                    x = F.relu(layer(x)) # -> output layer
                    hs.append(torch.flatten(F.interpolate(x, size=sampling_size), 2).to(device))

                else:
                    x = F.relu(layer(x))
                    hs.append(torch.flatten(F.interpolate(x, size=sampling_size), 2).to(device))

                if i % 2 == 0:
                    x = self.pooling_layer(x)

            x = torch.flatten(x, 1)

            for i, layer in enumerate(self.fc_layers, start=1):
                if i == len(self.fc_layers):
                    x = layer(x) # -> output layer
                    hs.append(x)
                else:
                    x = F.relu(layer(x))
                    hs.append(x)
                    x = self.dropout(x)

        else:
            if us is None:
                raise ValueError('us should be given')
            # conditional activation
            for i, layer in enumerate(self.conv_layers, start=1):
                us = us.squeeze()

                x = F.relu(layer(x)) * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1)
                idx += 1
                hs.append(x)

                if i % 2 == 0:
                    x = self.pooling_layer(x)

            x = torch.flatten(x, 1)
            # [TODO] masking?

            for i, layer in enumerate(self.fc_layers, start=1):
                if i == len(self.fc_layers): # -> output layer
                    x = layer(x) * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    hs.append(x)
                else:
                    x = F.relu(layer(x)) * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    idx += 1
                    hs.append(x)
                    x = self.dropout(x)

        return x, hs

class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, batch, conv_len, fc_len, adj_nodes, hidden_size = 64):
        super().__init__()
        self.conv1 = DenseSAGEConv(8*8, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(64,1, bias=False)
        self.minprob = minprob
        self.maxprob = maxprob
        self.conv_len = conv_len
        self.fc_len = fc_len

    def forward(self, hs, batch_adj):

        conv_hs = torch.cat(tuple(hs[i] for i in range(self.conv_len + 1)), dim=1)
        fc_hs = torch.cat(tuple(hs[i] for i in range(self.conv_len + 1, self.conv_len + self.fc_len + 1)), dim=1)
        fc_hs = fc_hs.unsqueeze(-1)
        fc_hs = fc_hs.expand(-1, -1, 64)

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
        u = torch.bernoulli(p).to(device) # [TODO] Error : probability -> not a number(nan), p is not in range of 0 to 1

        return u, p

def adj(model, bidirect = True, last_layer = True, edge2itself = True):
    if last_layer:
        num_nodes = (sum([layer.in_channels for layer in model.conv_layers]) + model.conv_layers[-1].out_channels) + (sum([layer.out_features for layer in model.fc_layers])) # conv 노드 + fc 노드
        nl = len(model.conv_layers) + len(model.fc_layers)
        conv_trainable_nodes = np.concatenate((np.ones(sum([layer.in_channels for layer in model.conv_layers])), np.zeros(model.conv_layers[-1].out_channels)), axis=0) # np.ones(sum([layer.in_channels for layer in model.conv_layers]))
        fc_trainable_nodes = np.ones(sum([layer.out_features for layer in model.fc_layers]))
        trainable_nodes = np.concatenate((conv_trainable_nodes, fc_trainable_nodes), axis=0)
        # trainable_nodes => [1,1,1,......,1,0,0,0] => input layer & hidden layer 의 노드 개수 = 1의 개수, output layer 의 노드 개수 = 0의 개수
    else:
        # num_nodes = sum([layer.in_channels for layer in model.conv_layers])
        # nl = len(model.conv_layers) - 1
        # trainable_nodes = np.ones(num_nodes)
        num_nodes = (sum([layer.in_channels for layer in model.conv_layers]) + model.conv_layers[-1].out__channels) + (model.fc_layers[0].out_features + model.fc_layers[1].out_features) # conv 노드 + fc 노드
        nl = len(model.conv_layers) + len(model.fc_layers) - 1
        conv_trainable_nodes = np.concatenate((np.ones(sum([layer.in_channels for layer in model.conv_layers])), np.zeros(model.conv_layers[-1].out_channels)), axis=0) # np.ones(sum([layer.in_channels for layer in model.conv_layers]))
        fc_trainable_nodes = np.ones(model.fc_layers[0].out_features + model.fc_layers[1].out_features)
        trainable_nodes = np.concatenate((conv_trainable_nodes, fc_trainable_nodes), axis=0)

    adjmatrix = np.zeros((num_nodes, num_nodes), dtype=np.int16)
    current_node = 0

    for i in range(nl):
        if i < len(model.conv_layers): # conv_layer
            layer = model.conv_layers[i]
            num_current = layer.in_channels
            num_next = layer.out_channels
        elif i == len(model.conv_layers): # conv_layer's output layer
            num_current = model.conv_layers[i - 1].out_channels
            num_next = model.fc_layers[i - len(model.conv_layers)].out_features
        elif i > len(model.conv_layers): # fc_layer
            layer = model.fc_layers[i - len(model.conv_layers)]
            num_current = layer.in_features
            num_next = layer.out_features


        for j in range(current_node, current_node + num_current):
            for k in range(current_node + num_current, current_node + num_current + num_next):
                adjmatrix[j, k] = 1

        # print start and end for j
        print(current_node, current_node + num_current)
        # print start and end for k
        print(current_node + num_current, current_node + num_current + num_next)
        print()
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
    args.add_argument('--BATCH_SIZE', type=int, default=60) # [TODO]: gradient accumulate step
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

    mlp_model = SimpleCNN().to(device)
    mlp_model.load_state_dict(torch.load('mlp_model_s=1.0_v=0.1_tau=0.32024-09-30_12-24-05.pt'))
    adj_, nodes_ = adj(mlp_model)
    adj_ = torch.stack([torch.Tensor(adj_) for _ in range(BATCH_SIZE)]).to(device)

    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, batch=BATCH_SIZE,
                     conv_len=len(mlp_model.conv_layers), fc_len=len(mlp_model.fc_layers), adj_nodes=len(nodes_), hidden_size=args.hidden_size).to(device)
    gnn_policy.load_state_dict(torch.load('gnn_policy_s=1.0_v=0.1_tau=0.32024-09-30_12-24-05.pt'))
    # model = Condnet_model(args=args.parse_args())

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

    wandb.init(project="condgtest",
                config=args.__dict__,
                name='condg_cnn_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau)
                )

    mlp_model.eval()
    gnn_policy.eval()

    specificity = []  # 결과 저장용 리스트
    # us_values = []
    us_variances = []

    for i, data in enumerate(tqdm(test_loader, 0)):
        # get batch
        inputs, labels = data
        # get batch

        inputs = inputs.to(device)
        outputs_1, hs = mlp_model(inputs)

        current_batch_size = hs[0].shape[0]

        if current_batch_size < BATCH_SIZE:
            adj_batch = adj_[:current_batch_size]
        else:
            adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

        # print(adj_batch.size())  # 크기 확인

        us, p = gnn_policy(hs, adj_batch)  # run gnn
        us_variances.append(np.var(us.detach().cpu().numpy()))
        outputs, hs = mlp_model(inputs, cond_drop=True, us=us.detach())

        # calculate accuracy
        pred = torch.argmax(outputs.to('cpu'), dim=1)
        acc_condg = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

        acc_random_shuffle = []
        for _ in range(100):
            # `us`를 랜덤하게 셔플링
            idx = torch.randperm(us.size(0))
            shuffled_us = us[idx]

            # 조건부 드롭을 적용하여 예측
            outputs, _ = mlp_model(inputs, cond_drop=True, us=shuffled_us)

            # 정확도 계산
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            acc_random_shuffle.append(acc)

        # 특정성 기록
        specificity.append([acc_condg, *acc_random_shuffle])

        # 특정성 결과를 numpy 배열로 변환
    specificity = np.array(specificity)

    # 평균 및 표준편차 계산
    accs_ = np.mean(specificity, axis=0)  # (101, )
    stds_ = np.std(specificity, axis=0)  # (101,)

    print("Accuracy with conditional GNN:", accs_[0])
    print("Mean accuracy with random shuffle:", accs_[1:].mean())
    print("Standard deviation with random shuffle:", stds_[1:].mean())

    # us_variances = [np.var(us_batch) for us_batch in us_values]  # 각 배치별로 `us`의 분산 계산
    # us_variance_mean = np.mean(us_variances)  # 모든 배치에 대한 평균 분산 계산
    # print("Variance of original `us` values:", us_variance_mean)

    us_variance_mean = np.mean(us_variances)
    print("Variance of original `us` values (average across batches):", us_variance_mean)

    wandb.finish()

if __name__ == '__main__':
    main()