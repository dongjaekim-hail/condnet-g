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
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 1024))
        self.layers.append(nn.Linear(1024, 1024))
        self.layers.append(nn.Linear(1024, 10))

    def forward(self, x, cond_drop=False, us=None):
        hs = [x]
        # flatten
        layer_cumsum = [0]
        for layer in self.layers:
            layer_cumsum.append(layer.in_features)
        layer_cumsum.append(self.layers[-1].out_features)
        layer_cumsum = np.cumsum(layer_cumsum)

        idx = 0
        if not cond_drop:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    # 첫 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    hs.append(x)
                elif i == 1:
                    # 두 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    hs.append(x)
                elif i == 2:
                    # 세 번째 레이어
                    x = layer(x)
                    hs.append(x)

        else:
            if us is None:
                raise ValueError('us should be given')
            # conditional activation
            for i, layer in enumerate(self.layers):
                us = us.squeeze()
                if i == 0:
                    # 첫 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    idx += 1
                elif i == 1:
                    # 두 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    idx += 1
                elif i == 2:
                    # 세 번째 레이어
                    x = layer(x)
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]

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

    def forward(self, hs, batch_adj):

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
def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=3.0)
    args.add_argument('--lambda_v', type=float, default=0.3)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--max_epochs', type=int, default=50)
    args.add_argument('--tau', type=float, default=0.3)
    args.add_argument('--condnet_min_prob', type=float, default=0.01)
    args.add_argument('--condnet_max_prob', type=float, default=0.99)
    args.add_argument('--learning_rate', type=float, default=0.05)
    args.add_argument('--BATCH_SIZE', type=int, default=256)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=64)
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
    num_inputs = 28**2

    mlp_model = Mlp().to(device)
    mlp_model.load_state_dict(torch.load('big_mlp_model_s=4.0_v=0.1_tau=0.3_lr=0.01.pt'))
    gnn_policy = Gnn(minprob=condnet_min_prob, maxprob=condnet_max_prob, hidden_size=args.hidden_size).to(device)
    gnn_policy.load_state_dict(torch.load('big_gnn_policy_s=4.0_v=0.1_tau=0.3_lr=0.01.pt'))

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

    wandb.init(project="condgtest",
                entity="hails",
                config=args.__dict__,
                name='big_condg_mlp_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + '_lr=' + str(args.learning_rate)
                )

    adj_, nodes_ = adj(mlp_model)
    adj_ = torch.stack([torch.Tensor(adj_) for _ in range(BATCH_SIZE)]).to(device)

    # 모델 평가 (테스트 데이터 사용)
    mlp_model.eval()
    gnn_policy.eval()

    specificity = []  # 결과 저장용 리스트
    # us_values = []
    us_variances = []
    for i, data in enumerate(test_loader, 0):

        inputs, labels = data
        # get batch

        inputs = inputs.view(-1, num_inputs).to(device)

        outputs_1, hs = mlp_model(inputs)
        hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
                       dim=1)  # changing dimension to 1 for putting hs vector in gnn
        hs = hs.detach()

        current_batch_size = hs.shape[0]
        if current_batch_size < BATCH_SIZE:
            adj_batch = adj_[:current_batch_size]
        else:
            adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

        us, p = gnn_policy(hs, adj_batch)  # run gnn
        # us_values.append(us.detach().cpu().numpy())
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

if __name__=='__main__':
    main()