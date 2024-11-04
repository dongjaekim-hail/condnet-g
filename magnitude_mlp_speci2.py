import torch
import torch.optim as optim
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import wandb
from datetime import datetime

wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")


class model_magnitude(nn.Module):
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

        self.mlp_nlayer = 0
        self.tau = args.tau

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.input_dim, mlp_hidden[0]))
        for i in range(nlayers):
            self.mlp.append(nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
        self.mlp.append(nn.Linear(mlp_hidden[i+1], output_dim))
        self.mlp.to(self.device)

        layer_cumsum = [0]
        for layer in self.mlp:
            layer_cumsum.append(layer.out_features)
        self.layer_cumsum = np.cumsum(layer_cumsum)

    def forward(self, x, cond_drop=False, us=None):
        if not cond_drop:
            h = x.view(-1, self.input_dim).to(self.device)
            us = []
            layer_masks = []
            for i in range(len(self.mlp) - 1):
                weight_magnitudes = torch.mean(self.mlp[i].weight.abs(), dim=1)

                sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)

                threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]

                mask = weight_magnitudes >= threshold

                mask = mask.repeat(x.size(0), 1)

                us.append(mask)

                h = F.relu(self.mlp[i](h)) * mask

                layer_masks.append(mask.float())

            weight_magnitudes = torch.mean(self.mlp[-1].weight.abs(), dim=1)

            sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)

            threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]

            mask = weight_magnitudes >= threshold

            mask = mask.repeat(x.size(0), 1)

            us.append(mask)

            h = self.mlp[-1](h)*mask

            layer_masks.append(mask.float())

            # softmax
            h = F.softmax(h, dim=1)
            layer_masks.append(mask.float())

            return h, us, layer_masks

        else:
            h = x.view(-1, self.input_dim).to(self.device)
            for i in range(len(self.mlp) - 1):

                h = F.relu(self.mlp[i](h)) * us[:, self.layer_cumsum[i]:self.layer_cumsum[i + 1]]

            h = self.mlp[-1](h) * us[:, self.layer_cumsum[2]:self.layer_cumsum[3]]

            # softmax
            h = F.softmax(h, dim=1)

            return h

def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=5)
    args.add_argument('--lambda_v', type=float, default=1e-2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=30)
    args.add_argument('--condnet_min_prob', type=float, default=1e-3)
    args.add_argument('--condnet_max_prob', type=float, default=1 - 1e-3)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=200)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args = args.parse_args()
    lambda_l2 = args.lambda_l2
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
                config=args.__dict__,
                name='out_runtime_weight_magnitude' + '_tau=' + str(args.tau) + '_' + dt_string
                )

    # create model
    model = model_magnitude(args)
    model.load_state_dict(torch.load('out_runtime_magnitude_weight_tau=0.62024-08-13_22-03-25.pt'))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.mlp.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    specificity = []  # 결과 저장용 리스트
    # us_values = []
    us_variances = []
    for i, data in enumerate(test_loader, 0):
        # get batch
        inputs, labels = data

        # get output
        outputs, us, layer_masks = model(inputs)
        us_mask = torch.cat(us, dim=1)
        us_variances.append(np.var(us_mask.detach().cpu().numpy()))

        # calculate accuracy
        pred = torch.argmax(outputs, dim=1).to('cpu')
        acc_mag = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

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
        specificity.append([acc_mag, *acc_random_shuffle])

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