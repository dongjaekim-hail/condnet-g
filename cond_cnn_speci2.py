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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleCNN(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_output_channels(self):  # todo add output channels when last conv
        channels = [0]
        conv_layers = [layer for layer in self.children() if isinstance(layer, nn.Conv2d)]
        fc_layers = [layer for layer in self.children() if isinstance(layer, nn.Linear)]

        # 모든 Conv2d 레이어의 in_channels 추가
        for layer in conv_layers:
            channels.append(layer.out_channels)

        # # 마지막 Conv2d 레이어의 out_channels 추가
        # if conv_layers:
        #     channels.append(conv_layers[-1].out_channels)

        # 모든 Linear 레이어의 out_features 추가
        for layer in fc_layers:
            channels.append(layer.out_features)

        return channels

class model_condnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.cnn = SimpleCNN().to(self.device)

        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob

        # HANDCRAFTED POLICY NET
        self.policy_net = nn.ModuleList()
        self.policy_net.append(nn.Linear(3 * 8 * 8, 64))
        self.policy_net.append(nn.Linear(64 * 8 * 8, 64))
        self.policy_net.append(nn.Linear(64 * 8 * 8, 128))
        self.policy_net.append(nn.Linear(128 * 8 * 8, 128))
        self.policy_net.append(nn.Linear(128 * 8 * 8, 256))
        self.policy_net.append(nn.Linear(256, 256))
        self.policy_net.append(nn.Linear(256, 10))

        self.policy_net.to(self.device)

        self.channels = self.cnn.get_output_channels()

    def forward(self, x, cond_drop=False, us=None):
        layer_cumsum = np.cumsum(self.channels)
        # policies = []
        # sample_probs = []
        # us = []
        # layer_masks = []
        if not cond_drop:
            policies = []
            sample_probs = []
            us = []
            layer_masks = []

            x = x.to(self.device)
            # u = torch.ones(x.shape[0], 64, x.shape[2], x.shape[3]).to(self.device)
            h = x
            u = torch.ones(h.shape[0], h.shape[1], h.shape[2], h.shape[3]).to(self.device)
            # 첫 번째 Conv 레이어와 마스킹
            # print(f"After first conv: {h.shape}")

            h_re = F.interpolate(h, size=(8, 8))
            p_i = self.policy_net[0](h_re.view(h_re.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)
            # u_i = u_i[:, channels_cumsum[0]:channels_cumsum[1], :, :]  # 채널을 맞추기 위해 차원을 추가
            # print(f"u_i after first conv: {u_i.shape}")
            # print(f"u after first conv: {u.shape}")
            # h = (h * u) * u_i
            h_next = F.relu(self.cnn.conv1(h * u)) * u_i
            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

            h_re = F.interpolate(h, size=(8, 8))
            p_i = self.policy_net[1](h_re.view(h_re.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)
            u_is = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16).to(self.device)
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)

            # 두 번째 Conv 레이어와 마스킹
            h_next = F.relu(self.cnn.conv2(h * u)) * u_i

            h = h_next
            u = u_is

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

            h = self.cnn.pool(h)

            h_re = F.interpolate(h, size=(8, 8))
            p_i = self.policy_net[2](h_re.view(h_re.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)
            # u = F.interpolate(u, size=(16, 16))

            # 두 번째 Conv 레이어와 마스킹
            h_next = F.relu(self.cnn.conv3(h * u)) * u_i

            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

            h_re = F.interpolate(h, size=(8, 8))
            p_i = self.policy_net[3](h_re.view(h_re.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)
            u_is = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8).to(self.device)
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)
            # u = F.interpolate(u, size=(16, 16))

            # 두 번째 Conv 레이어와 마스킹
            h_next = F.relu(self.cnn.conv4(h * u)) * u_i

            h = h_next
            u = u_is

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

            h = self.cnn.pool(h)

            # FC1 레이어와 마스킹
            u = u.reshape(u.size(0), -1)
            h = h.view(h.size(0), -1)

            p_i = self.policy_net[4](h.view(h.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)

            h_next = F.relu(self.cnn.fc1(h * u)) * u_i
            # print(f"After FC1: {h.shape}")

            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

            # FC2 레이어와 마스킹
            h = self.cnn.dropout(h)
            p_i = self.policy_net[5](h.view(h.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)

            h_next = F.relu(self.cnn.fc2(h * u)) * u_i
            # print(f"After FC1: {h.shape}")

            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)
            # print(f"After FC2: {h.shape}")

            # FC3 레이어와 마스킹
            h = self.cnn.dropout(h)
            p_i = self.policy_net[6](h.view(h.size(0), -1))
            p_i = F.sigmoid(p_i)
            p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            u_i = torch.bernoulli(p_i).to(self.device)

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
            # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
            #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
            us.append(u_i)

            h_next = F.relu(self.cnn.fc3(h * u)) * u_i
            # print(f"After FC1: {h.shape}")

            h = h_next

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)
            # print(f"After FC3: {h.shape}")

            return h, us, policies, sample_probs, layer_masks

        else:
            x = x.to(self.device)
            # u = torch.ones(x.shape[0], 64, x.shape[2], x.shape[3]).to(self.device)
            h = x
            u = torch.ones(h.shape[0], h.shape[1], h.shape[2], h.shape[3]).to(self.device)

            u_i = us[:, layer_cumsum[0]:layer_cumsum[1]]
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)

            h_next = F.relu(self.cnn.conv1(h * u)) * u_i
            h = h_next
            u = u_i

            u_i = us[:, layer_cumsum[1]:layer_cumsum[2]]
            u_is = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16).to(self.device)
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)

            # 두 번째 Conv 레이어와 마스킹
            h_next = F.relu(self.cnn.conv2(h * u)) * u_i
            h = h_next
            u = u_is

            h = self.cnn.pool(h)

            u_i = us[:, layer_cumsum[2]:layer_cumsum[3]]
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)

            # 두 번째 Conv 레이어와 마스킹
            h_next = F.relu(self.cnn.conv3(h * u)) * u_i
            h = h_next
            u = u_i

            u_i = us[:, layer_cumsum[3]:layer_cumsum[4]]
            u_is = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8).to(self.device)
            u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)

            # 두 번째 Conv 레이어와 마스킹
            h_next = F.relu(self.cnn.conv4(h * u)) * u_i
            h = h_next
            u = u_is

            h = self.cnn.pool(h)

            # FC1 레이어와 마스킹
            u = u.reshape(u.size(0), -1)
            h = h.view(h.size(0), -1)
            u_i = us[:, layer_cumsum[4]:layer_cumsum[5]]
            h_next = F.relu(self.cnn.fc1(h * u)) * u_i
            h = h_next
            u = u_i


            # FC2 레이어와 마스킹
            h = self.cnn.dropout(h)
            u_i = us[:, layer_cumsum[5]:layer_cumsum[6]]
            h_next = F.relu(self.cnn.fc2(h * u)) * u_i
            h = h_next
            u = u_i

            # FC3 레이어와 마스킹
            h = self.cnn.dropout(h)
            u_i = us[:, layer_cumsum[6]:layer_cumsum[7]]
            h_next = F.relu(self.cnn.fc3(h * u)) * u_i
            h = h_next

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
    args.add_argument('--tau', type=float, default=0.3)
    args.add_argument('--max_epochs', type=int, default=30)
    args.add_argument('--condnet_min_prob', type=float, default=1e-3)
    args.add_argument('--condnet_max_prob', type=float, default=1 - 1e-3)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--BATCH_SIZE', type=int, default=60)
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

    wandb.init(project="0.001cond_cnn_cifar10_edit",
                config=args.__dict__,
                name='0.3t0.001out_cond_cnn_cifar10_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau)
                )

    # create model
    model = model_condnet(args)
    model.load_state_dict(torch.load('0.3t0.001out_cond_cnn_cifar10_s=0.1_v=0.01_tau=0.32024-09-12_10-04-49.pt'))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.cnn.parameters():
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