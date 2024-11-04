import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
from tqdm import tqdm
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn.init as init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class model_condnet(nn.Module):
    def __init__(self,args):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.input_dim = 28*28
        mlp_hidden = [512, 256, 10]
        output_dim = mlp_hidden[-1]

        nlayers = args.nlayers
        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob

        self.mlp_nlayer = 0

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.input_dim, mlp_hidden[0]))
        for i in range(nlayers):
            self.mlp.append(nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
        self.mlp.append(nn.Linear(mlp_hidden[nlayers], output_dim))
        self.mlp.to(self.device)

        layer_cumsum = [0]
        for layer in self.mlp:
            layer_cumsum.append(layer.out_features)
        self.layer_cumsum = np.cumsum(layer_cumsum)

        # DOWNSAMPLE
        self.avg_poolings = nn.ModuleList()
        pool_hiddens = [512, *mlp_hidden]
        for i in range(len(self.mlp)):
            stride = round(pool_hiddens[i] / pool_hiddens[i + 1])
            self.avg_poolings.append(nn.AvgPool1d(kernel_size=stride, stride=stride))

        # UPSAMPLE
        self.upsample = nn.ModuleList()
        for i in range(len(self.mlp)):
            stride = round(pool_hiddens[i + 1] / 1024)
            self.upsample.append(nn.Upsample(scale_factor=stride, mode='nearest'))

        # # HANDCRAFTED POLICY NET
        n_each_policylayer = 1
        # n_each_policylayer = 1 # if you have only 1 layer perceptron for policy net
        self.policy_net = nn.ModuleList()
        temp = nn.ModuleList()
        # temp.append(nn.Linear(self.input_dim, mlp_hidden[0])) # BEFORE LARGE MODEL'S
        temp.append(nn.Linear(self.input_dim,  mlp_hidden[0]))
        self.policy_net.append(temp)

        for i in range(len(self.mlp) - 1):
            temp = nn.ModuleList()
            for j in range(n_each_policylayer):
                temp.append(nn.Linear(self.mlp[i].out_features, self.mlp[i+1].out_features)) # BEFORE LARGE MODEL'S
                # temp.append(nn.Linear(self.mlp[i].out_features, mlp_hidden[i]))
            self.policy_net.append(temp)
        self.policy_net.to(self.device)

    def forward(self, x, cond_drop=False, us=None):
        if not cond_drop:
            # return policies
            policies = []
            sample_probs = []
            us = []
            layer_masks = []
            x = x.view(-1, self.input_dim).to(self.device)

            h = x
            u = torch.ones(h.shape[0], h.shape[1]).to(self.device)

            for i in range(len(self.mlp)-1):

                h_clone = h.clone()
                p_is = self.policy_net[i][0](h_clone.detach())
                # p_i = self.policy_net[i][0](h)
                # Check for NaNs after first policy net layer
                if torch.isnan(p_is).any():
                    print(f"NaN detected in policy_net[{i}][0] output")

                p_i = F.sigmoid(p_is)

                # Check for NaNs after sigmoid activation
                if torch.isnan(p_i).any():
                    print(f"NaN detected after sigmoid(policy_net[{i}][0] output)")

                for j in range(1, len(self.policy_net[i])):
                    p_is = self.policy_net[i][j](p_i)
                    p_i = F.sigmoid(p_is)

                # p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
                p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)

                if np.any(np.isnan(p_i.cpu().detach().numpy())):
                    print('wait a sec')

                u_i = torch.bernoulli(p_i).to(self.device)

                # debug[TODO]
                # u_i = torch.ones(u_i.shape[0], u_i.shape[1])

                if u_i.sum() == 0:
                    idx = np.random.uniform(0, u_i.shape[0], size = (1)).astype(np.int16)
                    u_i[idx] = 1

                sampling_prob = p_i * u_i + (1-p_i) * (1-u_i)

                # idx = torch.where(u_i == 0)[0]

                # h_next = F.relu(self.mlp[i](h*u.detach()))*u_i
                us.append(u_i)
                h_next = F.relu(self.mlp[i](h*u))*u_i
                h = h_next
                u = u_i

                policies.append(p_i)
                sample_probs.append(sampling_prob)
                layer_masks.append(u_i)

            h_clone = h.clone()
            p_is = self.policy_net[2][0](h_clone.detach())
            # p_i = self.policy_net[i][0](h)
            # Check for NaNs after first policy net layer
            if torch.isnan(p_is).any():
                print(f"NaN detected in policy_net[{2}][0] output")

            p_i = F.sigmoid(p_is)

            # Check for NaNs after sigmoid activation
            if torch.isnan(p_i).any():
                print(f"NaN detected after sigmoid(policy_net[{2}][0] output)")

            for j in range(1, len(self.policy_net[2])):
                p_is = self.policy_net[2][j](p_i)
                p_i = F.sigmoid(p_is)

            # p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)

            if np.any(np.isnan(p_i.cpu().detach().numpy())):
                print('wait a sec')

            u_i = torch.bernoulli(p_i).to(self.device)

            # debug[TODO]
            # u_i = torch.ones(u_i.shape[0], u_i.shape[1])

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            us.append(u_i)
            h_next = (self.mlp[-1](h*u))*u_i
            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

            h = F.softmax(h, dim=1)

            return h, us, policies, sample_probs, layer_masks

        else:
            x = x.view(-1, self.input_dim).to(self.device)

            h = x
            u = torch.ones(h.shape[0], h.shape[1]).to(self.device)

            for i in range(len(self.mlp) - 1):
                u_i = us[:, self.layer_cumsum[i]:self.layer_cumsum[i + 1]]
                h_next = F.relu(self.mlp[i](h * u)) * u_i
                h = h_next
                u = u_i

            u_i = us[:, self.layer_cumsum[2]:self.layer_cumsum[3]]
            h_next = (self.mlp[-1](h * u)) * u_i
            h = h_next

            h = F.softmax(h, dim=1)

            return h


def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=0.5)
    args.add_argument('--lambda_v', type=float, default=0.01)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=0.05)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=30)
    args.add_argument('--condnet_min_prob', type=float, default=1e-3)
    args.add_argument('--condnet_max_prob', type=float, default=1 - 1e-3)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--BATCH_SIZE', type=int, default=256)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=32)
    args = args.parse_args()
    lambda_s = args.lambda_s
    lambda_v = args.lambda_v
    lambda_l2 = args.lambda_l2
    lambda_pg = args.lambda_pg
    tau = args.tau
    learning_rate = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE


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
                name='0.001out_cond_mlp_mnist_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau)
                )

    # create model
    model = model_condnet(args)
    model.load_state_dict(torch.load('0.001output_cond_s=0.5_v=0.01_tau=0.62024-09-05_17-27-13.pt'))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.mlp.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.policy_net.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    specificity = []  # 결과 저장용 리스트
    # us_values = []
    us_variances = []
    for i, data in enumerate(tqdm(test_loader, 0)):
        # get batch
        inputs, labels = data

        # get output
        outputs, us, policies, sample_probs, layer_masks = model(inputs)

        us_mask = torch.cat(us, dim=1)
        us_variances.append(np.var(us_mask.detach().cpu().numpy()))

        # calculate accuracy
        pred = torch.argmax(outputs, dim=1).to('cpu')
        acc_cond = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

        acc_random_shuffle = []
        for _ in range(100):
            # `us`를 랜덤하게 셔플링
            idx = torch.randperm(us_mask.size(0))
            shuffled_us = us_mask[idx]

            # 조건부 드롭을 적용하여 예측
            outputs = model(inputs, cond_drop=True, us=shuffled_us)

            # 정확도 계산
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            acc_random_shuffle.append(acc)

        # 특정성 기록
        specificity.append([acc_cond, *acc_random_shuffle])

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