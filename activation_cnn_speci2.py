import torch
import torch.optim as optim
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import wandb
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

class SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SimpleCNN, self).__init__()
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

        self.tau = args.tau

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.fc_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x, cond_drop=False, us=None):
        layer_cumsum = [0]
        for layer in self.conv_layers:
            layer_cumsum.append(layer.out_channels)
        for layer in self.fc_layers:
            layer_cumsum.append(layer.out_features)
        layer_cumsum = np.cumsum(layer_cumsum)

        if not cond_drop:
            us = []
            layer_masks = []

            # Conv1
            x = F.relu(self.conv1(x))
            activation_magnitudes = torch.norm(torch.norm(x, dim=(2, 3)), dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(sorted_activation_magnitudes.size(0) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            mask = mask.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * mask.float()
            layer_masks.append(mask.float())

            # Conv2
            x = F.relu(self.conv2(x))
            activation_magnitudes = torch.norm(torch.norm(x, dim=(2, 3)), dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(sorted_activation_magnitudes.size(0) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            mask = mask.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * mask.float()
            x = self.pool(x)
            layer_masks.append(mask.float())

            # Conv3
            x = F.relu(self.conv3(x))
            activation_magnitudes = torch.norm(torch.norm(x, dim=(2, 3)), dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(sorted_activation_magnitudes.size(0) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            mask = mask.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * mask.float()
            layer_masks.append(mask.float())

            # Conv4
            x = F.relu(self.conv4(x))
            activation_magnitudes = torch.norm(torch.norm(x, dim=(2, 3)), dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(sorted_activation_magnitudes.size(0) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            mask = mask.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * mask.float()
            x = self.pool(x)
            layer_masks.append(mask.float())

            x = x.view(-1, 128 * 8 * 8)

            # FC1
            x = F.relu(self.fc1(x))
            activation_magnitudes = torch.norm(x, dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(len(sorted_activation_magnitudes) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            x = x * mask
            x = self.dropout(x)
            layer_masks.append(mask.float())

            # FC2
            x = F.relu(self.fc2(x))
            activation_magnitudes = torch.norm(x, dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(len(sorted_activation_magnitudes) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            x = x * mask
            x = self.dropout(x)
            layer_masks.append(mask.float())

            # FC3
            x = self.fc3(x)
            activation_magnitudes = torch.norm(x, dim=1)
            sorted_activation_magnitudes, _ = torch.sort(activation_magnitudes, descending=True)
            threshold = sorted_activation_magnitudes[round(len(sorted_activation_magnitudes) * self.tau)]
            mask = activation_magnitudes >= threshold
            mask = mask.unsqueeze(1).expand(-1, x.size(1))
            us.append(mask)
            x = x * mask
            layer_masks.append(mask.float())

            return x, us, layer_masks
        else:
            # Conv1
            x = F.relu(self.conv1(x))
            uss = us[:, layer_cumsum[0]:layer_cumsum[1]]
            uss = uss.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * uss

            # Conv2
            x = F.relu(self.conv2(x))
            uss = us[:, layer_cumsum[1]:layer_cumsum[2]]
            uss = uss.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * uss
            x = self.pool(x)

            # Conv3
            x = F.relu(self.conv3(x))
            uss = us[:, layer_cumsum[2]:layer_cumsum[3]]
            uss = uss.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * uss

            # Conv4
            x = F.relu(self.conv4(x))
            uss = us[:, layer_cumsum[3]:layer_cumsum[4]]
            uss = uss.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * uss
            x = self.pool(x)

            x = x.view(-1, 128 * 8 * 8)

            # FC1
            x = F.relu(self.fc1(x))
            uss = us[:, layer_cumsum[4]:layer_cumsum[5]]
            x = x * uss
            x = self.dropout(x)

            # FC2
            x = F.relu(self.fc2(x))
            uss = us[:, layer_cumsum[5]:layer_cumsum[6]]
            x = x * uss
            x = self.dropout(x)

            # FC3
            x = self.fc3(x)
            uss = us[:, layer_cumsum[6]:layer_cumsum[7]]
            x = x * uss

            return x

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
    args.add_argument('--lambda_pg', type=float, default=0.05)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=30)
    args.add_argument('--condnet_min_prob', type=float, default=1e-3)
    args.add_argument('--condnet_max_prob', type=float, default=1 - 1e-3)
    args.add_argument('--lr', type=float, default=0.0003)
    args.add_argument('--BATCH_SIZE', type=int, default=60)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=32)
    args = args.parse_args()
    lambda_l2 = args.lambda_l2
    tau = args.tau
    learning_rate = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE


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
                name='out_cnn_runtime_activation_magnitude' + '_tau=' + str(args.tau) + '_' + dt_string
                )

    # create model
    model = SimpleCNN(args)
    model.load_state_dict(torch.load('out_cnn_runtime_activation_magnitude_tau=0.62024-11-05_08-45-59.pt'))

    if torch.cuda.is_available():
        model = model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    specificity = []  # 결과 저장용 리스트
    # us_values = []
    us_variances = []
    for i, data in enumerate(test_loader, 0):
        # get batch
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # get output
        outputs, us, layer_masks = model(inputs)
        us_mask = torch.cat(us, dim=1)
        us_variances.append(np.var(us_mask.detach().cpu().numpy()))

        # calculate accuracy
        pred = torch.argmax(outputs, dim=1)
        acc_mag = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
        acc_random_shuffle = []
        for _ in range(100):
            # `us`를 랜덤하게 셔플링
            idx = torch.randperm(us_mask.size(0))
            shuffled_us = us_mask[idx]

            # 조건부 드롭을 적용하여 예측
            outputs = model(inputs, cond_drop=True, us=shuffled_us)

            # 정확도 계산
            pred = torch.argmax(outputs.to(device), dim=1)
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