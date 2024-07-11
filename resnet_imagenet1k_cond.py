import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

from datetime import datetime


class model_condnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.resnet = models.resnet50(pretrained=True).to(self.device)

        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob

        # # HANDCRAFTED POLICY NET
        # self.policy_net = nn.ModuleList()
        # self.policy_net.append(nn.Linear(64 * 112 * 112, 1024))  # conv1 출력 특징
        # self.policy_net.append(nn.Linear(64 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(64 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(256 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(64 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(64 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(256 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(64 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(64 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(256 * 56 * 56, 1024))  # layer1의 첫 번째 블록의 conv3 출력 특징
        #
        #
        # self.policy_net.append(nn.Linear(128 * 56 * 56, 1024))  # layer2의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(128 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(512 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(128 * 56 * 56, 1024))  # layer2의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(128 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(512 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(128 * 56 * 56, 1024))  # layer2의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(128 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(512 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(128 * 56 * 56, 1024))  # layer2의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(128 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(512 * 28 * 28, 1024))  # layer2의 첫 번째 블록의 conv3 출력 특징
        #
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(1024 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(1024 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(1024 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(1024 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(1024 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(256 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(1024 * 14 * 14, 1024))  # layer3의 첫 번째 블록의 conv3 출력 특징
        #
        #
        # self.policy_net.append(nn.Linear(512 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(512 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(2048 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(512 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(512 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(2048 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv3 출력 특징
        # self.policy_net.append(nn.Linear(512 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv1 출력 특징
        # self.policy_net.append(nn.Linear(512 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv2 출력 특징
        # self.policy_net.append(nn.Linear(2048 * 7 * 7, 1024))  # layer4의 첫 번째 블록의 conv3 출력 특징

        # HANDCRAFTED POLICY NET
        self.policy_net = nn.ModuleList()
        self.policy_net.append(nn.Linear(64 * 112 * 112, 16))  # conv1 출력 특징
        self.policy_net.append(nn.Linear(64 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(64 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(256 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(64 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(64 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(256 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(64 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(64 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(256 * 56 * 56, 16))  # layer1의 첫 번째 블록의 conv3 출력 특징

        self.policy_net.append(nn.Linear(128 * 56 * 56, 16))  # layer2의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(512 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(512 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(512 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(128 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(512 * 28 * 28, 16))  # layer2의 첫 번째 블록의 conv3 출력 특징

        self.policy_net.append(nn.Linear(256 * 28 * 28, 16))  # layer3의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(1024 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(1024 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(1024 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(1024 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(1024 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(256 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(1024 * 14 * 14, 16))  # layer3의 첫 번째 블록의 conv3 출력 특징

        self.policy_net.append(nn.Linear(512 * 14 * 14, 16))  # layer4의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(512 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(2048 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(512 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(512 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(2048 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv3 출력 특징
        self.policy_net.append(nn.Linear(512 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv1 출력 특징
        self.policy_net.append(nn.Linear(512 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv2 출력 특징
        self.policy_net.append(nn.Linear(2048 * 7 * 7, 16))  # layer4의 첫 번째 블록의 conv3 출력 특징

        self.policy_net.to(self.device)

    def forward(self, x):
        policies = []
        sample_probs = []
        layer_masks = []

        x = x.to(self.device)
        u = torch.ones(x.shape[0], 64, x.shape[2] // 2, x.shape[3] // 2).to(self.device)

        # ResNet50의 첫 번째 레이어
        h = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        # print(f"conv1 output size: {h.size()}")
        p_i = self.policy_net[0](h.view(h.size(0), -1))
        p_i = torch.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)
        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1
        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
                            align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
        h = (h * u) * u_i
        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        u = u_i

        h = self.resnet.maxpool(h)
        # print(f"maxpool output size: {h.size()}")

        policy_index = 1
        # 각 레이어의 bottleneck 블록을 통과한 특징
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for bottleneck in layer:
                residual = h

                out = F.relu(bottleneck.bn1(bottleneck.conv1(h)))
                # print(f"layer bottleneck.conv1 output size: {out.size()}")
                p_i = self.policy_net[policy_index](out.view(out.size(0), -1))
                policy_index += 1
                p_i = torch.sigmoid(p_i)
                p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
                u_i = torch.bernoulli(p_i).to(self.device)
                if u_i.sum() == 0:
                    idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                    u_i[idx] = 1
                sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
                u_i = F.interpolate(u_i.unsqueeze(1), size=out.size(1) * out.size(2) * out.size(3), mode='linear',
                                    align_corners=True).squeeze(1).view(out.size(0), out.size(1), out.size(2), out.size(3))
                u = F.interpolate(u.unsqueeze(1), size=(out.size(1), out.size(2), out.size(3)), mode='nearest').squeeze(1)
                out = (out * u) * u_i

                policies.append(p_i)
                sample_probs.append(sampling_prob)
                layer_masks.append(u_i)

                u = u_i

                out = F.relu(bottleneck.bn2(bottleneck.conv2(out)))
                # print(f"layer bottleneck.conv2 output size: {out.size()}")
                p_i = self.policy_net[policy_index](out.view(out.size(0), -1))
                policy_index += 1
                p_i = torch.sigmoid(p_i)
                p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
                u_i = torch.bernoulli(p_i).to(self.device)
                if u_i.sum() == 0:
                    idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                    u_i[idx] = 1
                sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
                u_i = F.interpolate(u_i.unsqueeze(1), size=out.size(1) * out.size(2) * out.size(3), mode='linear',
                                    align_corners=True).squeeze(1).view(out.size(0), out.size(1), out.size(2), out.size(3))
                u = F.interpolate(u.unsqueeze(1), size=(out.size(1), out.size(2), out.size(3)), mode='nearest').squeeze(1)
                out = (out * u) * u_i

                policies.append(p_i)
                sample_probs.append(sampling_prob)
                layer_masks.append(u_i)

                u = u_i

                out = bottleneck.bn3(bottleneck.conv3(out))
                # print(f"layer bottleneck.conv3 output size: {out.size()}")
                if bottleneck.downsample is not None:
                    residual = bottleneck.downsample(h)
                out += residual
                out = bottleneck.relu(out)
                # print(f"layer bottleneck output size: {out.size()}")
                p_i = self.policy_net[policy_index](out.view(out.size(0), -1))
                policy_index += 1
                p_i = torch.sigmoid(p_i)
                p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
                u_i = torch.bernoulli(p_i).to(self.device)
                if u_i.sum() == 0:
                    idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                    u_i[idx] = 1
                sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
                u_i = F.interpolate(u_i.unsqueeze(1), size=out.size(1) * out.size(2) * out.size(3), mode='linear',
                                    align_corners=True).squeeze(1).view(out.size(0), out.size(1), out.size(2), out.size(3))
                # print(f"layer1 u_i size: {u_i.size()}")
                u = F.interpolate(u.unsqueeze(1), size=(out.size(1), out.size(2), out.size(3)), mode='nearest').squeeze(1)
                # print(f"layer1 u size: {u.size()}")
                out = (out * u) * u_i

                policies.append(p_i)
                sample_probs.append(sampling_prob)
                layer_masks.append(u_i)

                u = u_i

                h = out

        h = self.resnet.avgpool(h)
        h = torch.flatten(h, 1)
        h = self.resnet.fc(h)

        return h, policies, sample_probs, layer_masks

def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=7)
    args.add_argument('--lambda_v', type=float, default=1.2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=200)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--BATCH_SIZE', type=int, default=2)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args = args.parse_args()
    lambda_s = args.lambda_s
    lambda_v = args.lambda_v
    lambda_l2 = args.lambda_l2
    lambda_pg = args.lambda_pg
    tau = args.tau
    learning_rate = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE

    dataset_path = r'C:\Users\97dnd\anaconda3\envs\torch\pr\resnet\data'
    dataset = load_from_disk(dataset_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    wandb.init(project="condgnet",
                config=args.__dict__,
                name='resnet_cond_imagenet1k_conv' + '_tau=' + str(args.tau)
                )

    # create model
    model = model_condnet(args)
    # model = model_condnet2()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.resnet.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.policy_net.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)
    # mlp_optimizer = optim.SGD(model.mlp.parameters(), lr=learning_rate,
    #                       momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(model.policy_net.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=lambda_l2)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0
        PGs = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(train_loader, 0):
            mlp_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고

            # 순전파 + 역전파 + 최적화를 한 후
            outputs, policies, sample_probs, layer_masks  = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 1000)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels.to(model.device))
            # Compute the regularization loss L

            L = c + lambda_s * (torch.pow(torch.stack(policies).mean(axis=1) - torch.tensor(tau).to(model.device), 2).mean() +
                                torch.pow(torch.stack(policies).mean(axis=2) - torch.tensor(tau).to(model.device), 2).mean())

            L += lambda_v * (-1) * (torch.stack(policies).to('cpu').var(axis=1).mean() +
                                    torch.stack(policies).to('cpu').var(axis=2).mean())



            # Compute the policy gradient (PG) loss
            logp = torch.log(torch.cat(policies)).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

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

            # wandb log training/batch
            wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': PG.item(), 'train/batch_tau': tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, np.mean([tau_.mean().item() for tau_ in layer_masks])
                                                                                                     ))

        # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': tau, 'train/epoch_PG': PGs/bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0
        PGs = 0

        model.eval()
        with torch.no_grad():
            # calculate accuracy on test set
            acc = 0
            bn = 0
            taus = 0
            for i, data in enumerate(test_loader, 0):
                bn += 1
                # get batch
                inputs, labels = data

                # make one hot vector
                y_batch_one_hot = torch.zeros(labels.shape[0], 1000)
                y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1,).tolist()] = 1

                # get output
                outputs, policies, sample_probs, layer_masks = model(torch.tensor(inputs))

                # calculate accuracy
                pred = torch.argmax(outputs, dim=1).to('cpu')
                acc  = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                c = C(outputs, labels.to(model.device))

                # Compute the regularization loss L

                L = c + lambda_s * (torch.pow(torch.stack(policies).mean(axis=1) - torch.tensor(tau).to(model.device), 2).mean() +
                                    torch.pow(torch.stack(policies).mean(axis=2) - torch.tensor(tau).to(model.device), 2).mean())

                L += lambda_v * (-1) * (torch.stack(policies).var(axis=1).mean() +
                                        torch.stack(policies).var(axis=2).mean())



                # Compute the policy gradient (PG) loss
                logp = torch.log(torch.cat(policies)).sum(axis=1).mean()
                PG = lambda_pg * c * (-logp) + L

                # wandb log test/batch
                # wandb.log({'test/batch_acc': acc, 'test/batch_cost': c.to('cpu').item(), 'test/batch_pg': PG.to('cpu').item()})

                # addup loss and acc
                costs += c.to('cpu').item()
                accs += acc
                PGs += PG.to('cpu').item()

                tau_ = torch.stack(policies).mean().detach().item()
                taus += tau_
            #print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log test/epoch
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn, 'test/epoch_pg': PGs / bn, 'test/epoch_tau': taus / bn })
        torch.save(model.state_dict(), './resnet_cond_imagenet1k_conv' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
    wandb.finish()
if __name__=='__main__':
    main()