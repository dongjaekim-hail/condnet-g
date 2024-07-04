import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional import accuracy
from datasets import load_from_disk
import torchvision.models as models
from datetime import datetime
import wandb
import numpy as np
from torch_geometric.nn import DenseSAGEConv
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, debug: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.debug = debug
    def __len__(self):
        return len(self.train_dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item['image'], item['label']
        if self.transform:
            image = self.transform(image)
        return image, label
    def setup(self, stage=None):
        class ImagenetDataset(Dataset):
            def __init__(self, dataset_path, transform=self.transform):
                self.dataset = load_from_disk(dataset_path)
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]
                image, label = item['image'], item['label']
                if self.transform:
                    image = self.transform(image)
                return image, label

        if stage == 'fit' or stage is None:
            if self.debug:
                self.train_dataset = ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)
            else:
                self.train_dataset = ImagenetDataset(os.path.join(self.data_dir, 'train'), transform=self.transform)
            self.val_dataset = ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)
        elif stage == 'test' or stage == 'predict':
            self.test_dataset = ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)
            self.predict_dataset = ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)

class ImagenetDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset = load_from_disk(dataset_path)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item['image'], item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

class ResNet50(torch.nn.Module):
    def __init__(self, device):
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
        self.device = device

    def forward(self, x, cond_drop=False, us=None, channels=None):
        hs = [torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(self.device)]
        if not cond_drop:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            hs.append(torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(self.device))
            x = self.max_pool(x)
            count = 0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=(7, 7)), 2).to(self.device))
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=(7, 7)), 2).to(self.device))
                    out = bottleneck.conv3(out)
                    out = bottleneck.bn3(out)
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                    out += residual
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=(7, 7)), 2).to(self.device))
                    x = out
                count += 1
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
            i = 0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out)
                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1
                    hs.append(out)
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    out = bottleneck.relu(out)
                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1
                    hs.append(out)
                    out = bottleneck.conv3(out)
                    out = bottleneck.bn3(out)
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                    out += residual
                    out = bottleneck.relu(out)
                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1
                    hs.append(out)
                    x = out

            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x, hs

def adj(resnet_model, bidirect=True, last_layer=True, edge2itself=True):
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
    adjmatrix[adjmatrix != 0] = 1
    return adjmatrix, num_channels_ls

class runtime_pruner(L.LightningModule):
    def __init__(self, resnet, gnn_policy, adj_, num_channels_ls, args, device):
        super().__init__()
        self.resnet = resnet
        self.gnn_policy = gnn_policy
        self.adj = adj_
        self.num_channels_ls = num_channels_ls
        self.tau = args.tau
        self.lambda_s = args.lambda_s
        self.lambda_v = args.lambda_v
        self.lambda_l2 = args.lambda_l2
        self.lambda_pg = args.lambda_pg
        self.learning_rate = args.learning_rate
        self.accum_step = args.accum_step
        self.automatic_optimization = False

        self.e_time_resnet_eval = []
        self.e_time_gnn_forward = []
        self.e_time_resnet_pruned_forward = []
        self.e_time_loss_backward = []
        self.e_time_resnet_opt_step = []
        self.e_time_gnn_opt_step = []

    def training_step(self, batch, batch_idx):
        # resnet_opt, gnn_opt = self.optimizers()
        # resnet_opt.zero_grad()
        gnn_opt = self.optimizers()
        gnn_opt.zero_grad()
        # if self.current_epoch % 2 ==0:
        #     # resnet 만 학습
        # else:
        #     # learn both

        # split batch with self.accum_step
        c_ = 0
        acc_ = 0
        acc_bf = 0
        PG_ = 0
        tau_ = 0
        L_ = 0

        x, y = batch
        accum_batch_size = len(x) // self.accum_step
        for i in range(0, len(x), accum_batch_size):
            x_batch = x[i:i + accum_batch_size]
            y_batch = y[i:i + accum_batch_size]
            # torch.cuda.synchronize()
            # time_ = time.time()
            self.resnet.eval()
            with torch.no_grad():
                y_hat_surrogate, hs = self.resnet(x_batch)
                acc_bf = accuracy(torch.argmax(y_hat_surrogate, dim=1), y_batch, task='multiclass', num_classes=1000)
            # torch.cuda_syncrhonize()
            # etime = time.time() - time_

            us, p = self.gnn_policy(hs, self.adj)
            # self.resnet.train()

            outputs, hs = self.resnet(x_batch, cond_drop=True, us=us.detach(), channels=self.num_channels_ls)
            c = F.cross_entropy(outputs, y_batch)
            try:
                acc = accuracy(torch.argmax(outputs, dim=1), y_batch, task='multiclass', num_classes=1000)
            except:
                print('')
            L = c + self.lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(self.tau), 2).mean() +
                                     torch.pow(p.squeeze().mean(axis=1) - torch.tensor(self.tau), 2).mean())
            L += self.lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                         p.squeeze().var(axis=1).mean())
            logp = torch.log(p.squeeze()).sum(axis=1).mean()
            PG = self.lambda_pg * c * (-logp) + L
            PG /= self.accum_step
            self.manual_backward(PG)

            tau__ = us.mean().detach().item()

            c_ += c.item()
            acc_ += acc.item()
            PG_ += PG.item()
            L_ += L.item()
            tau_ += tau__

        # resnet_opt.step()
        gnn_opt.step()

        c = c_ / self.accum_step
        acc = acc_ / self.accum_step
        PG = PG_ / self.accum_step
        L = L_ / self.accum_step
        tau = tau_ / self.accum_step

        self.log('train/batch_cost', c, prog_bar=True)
        self.log('train/batch_acc', acc, prog_bar=True)
        self.log('train/batch_acc_bf', acc_bf, prog_bar=True)
        self.log('train/batch_pg', PG, prog_bar=True)
        self.log('train/batch_loss', L, prog_bar=True)
        self.log('train/batch_tau', tau, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        c_ = 0
        acc_ = 0
        acc_bf_ = 0
        PG_ = 0
        L_ = 0
        tau_ = 0

        accum_batch_size = len(x) // self.accum_step
        for i in range(0, len(x), accum_batch_size):
            x_batch = x[i:i + accum_batch_size]
            y_batch = y[i:i + accum_batch_size]

            self.resnet.eval()
            with torch.no_grad():
                y_hat_surrogate, hs = self.resnet(x_batch)
                acc_bf = accuracy(torch.argmax(y_hat_surrogate, dim=1), y_batch, task='multiclass', num_classes=1000)
                us, p = self.gnn_policy(hs, self.adj)

                # if torch.isnan(p).any() or torch.isinf(p).any() or (p < 0).any() or (p > 1).any():
                #     print("Invalid p values detected during validation")
                #     print(f"p min: {p.min().item()}, p max: {p.max().item()}")
                #     raise ValueError('Invalid p values detected during validation')
                #
                #     # Clipping p to ensure it's in the valid range [0, 1]
                # p = torch.clamp(p, min=0.0, max=1.0)

                outputs, hs = self.resnet(x_batch, cond_drop=True, us=us.detach(), channels=self.num_channels_ls)
                # c = F.cross_entropy(F.softmax(outputs, dim=1), y_batch)
                c = F.cross_entropy(outputs, y_batch)
                acc = accuracy(torch.argmax(outputs, dim=1), y_batch, task='multiclass', num_classes=1000)
                L = c + self.lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(self.tau), 2).mean() +
                                         torch.pow(p.squeeze().mean(axis=1) - torch.tensor(self.tau), 2).mean())
                L += self.lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                             p.squeeze().var(axis=1).mean())
                logp = torch.log(p.squeeze()).sum(axis=1).mean()
                # logp = torch.log(p.squeeze() + 1e-10).sum(axis=1).mean()  # Avoid log(0)
                PG = self.lambda_pg * c * (-logp) + L
                PG /= self.accum_step

                tau__ = us.mean().detach().item()

            c_ += c.item()
            acc_ += acc.item()
            PG_ += PG.item()
            L_ += L.item()
            tau_ += tau__

        c = c_ / self.accum_step
        acc = acc_ / self.accum_step
        acc_bf = acc_bf_ / self.accum_step
        PG = PG_ / self.accum_step
        L = L_ / self.accum_step
        tau = tau_ / self.accum_step

        self.log('test/batch_cost', c, prog_bar=True)
        self.log('test/batch_acc', acc, prog_bar=True)
        self.log('test/batch_acc_bf', acc_bf, prog_bar=True)
        self.log('test/batch_pg', PG, prog_bar=True)
        self.log('test/batch_loss', L, prog_bar=True)
        self.log('test/batch_tau', tau, prog_bar=True)

    def configure_optimizers(self):
        # resnet_optimizer = torch.optim.SGD(self.resnet.parameters(), lr=self.learning_rate,
        #                                    momentum=0.9, weight_decay=self.lambda_l2)
        gnn_optimizer = torch.optim.SGD(self.gnn_policy.parameters(), lr=self.learning_rate,
                                        momentum=0.9, weight_decay=self.lambda_l2)
        return gnn_optimizer


# class Gnn(nn.Module):
#     def __init__(self, minprob, maxprob, hidden_size=64, device='cpu'):
#         super().__init__()
#         self.conv1 = DenseSAGEConv(7 * 7, hidden_size)
#         self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
#         self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
#         self.fc1 = nn.Linear(hidden_size, 1)
#         self.minprob = minprob
#         self.maxprob = maxprob
#         self.device = device
#
#     def forward(self, hs, adj):
#         device = self.device
#         h = hs[0]
#         for i in hs[1:]:
#             h = torch.cat((h, i), dim=1)
#         h = h.to(device)
#
#         batch_adj = torch.stack([torch.Tensor(adj) for _ in range(h.shape[0])])
#         batch_adj = batch_adj.to(device)
#
#         print(f"Input h: min {h.min().item()}, max {h.max().item()}")
#         if torch.isnan(h).any() or torch.isinf(h).any():
#             raise ValueError('Invalid values detected in input h')
#
#         # 정규화
#         h = (h - h.min()) / (h.max() - h.min())
#
#         print(f"Normalized h: min {h.min().item()}, max {h.max().item()}")
#
#         print(f"Input adj: min {batch_adj.min().item()}, max {batch_adj.max().item()}")
#         if torch.isnan(batch_adj).any() or torch.isinf(batch_adj).any():
#             raise ValueError('Invalid values detected in input adj')
#
#         hs = torch.sigmoid(self.conv1(h, batch_adj))
#         print(f"After conv1: min {hs.min().item()}, max {hs.max().item()}")
#         if torch.isnan(hs).any() or torch.isinf(hs).any():
#             raise ValueError('Invalid values detected after conv1')
#
#         hs = torch.sigmoid(self.conv2(hs, batch_adj))
#         print(f"After conv2: min {hs.min().item()}, max {hs.max().item()}")
#         if torch.isnan(hs).any() or torch.isinf(hs).any():
#             raise ValueError('Invalid values detected after conv2')
#
#         hs = torch.sigmoid(self.conv3(hs, batch_adj))
#         print(f"After conv3: min {hs.min().item()}, max {hs.max().item()}")
#         if torch.isnan(hs).any() or torch.isinf(hs).any():
#             raise ValueError('Invalid values detected after conv3')
#
#         hs = self.fc1(hs)
#         print(f"After fc1: min {hs.min().item()}, max {hs.max().item()}")
#         if torch.isnan(hs).any() or torch.isinf(hs).any():
#             raise ValueError('Invalid values detected after fc1')
#
#         p = torch.sigmoid(hs)
#         p = p * (self.maxprob - self.minprob) + self.minprob
#
#         # Clipping p to ensure it's in the valid range [0, 1]
#         p = torch.clamp(p, min=0.0, max=1.0)
#         print(f"p min: {p.min().item()}, p max: {p.max().item()}")
#         if torch.isnan(p).any() or torch.isinf(p).any():
#             raise ValueError('Invalid p values detected')
#
#         u = torch.bernoulli(p).to(device)
#
#         return u, p


class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, hidden_size = 64, device ='cpu'):
        super().__init__()
        self.conv1 = DenseSAGEConv(7*7, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.minprob = minprob
        self.maxprob = maxprob
        self.device = device

    def forward(self, hs, adj):
        device = self.device
        h = hs[0]
        for i in hs[1:]:
            h = torch.cat((h, i), dim=1)
        h = h.to(device)

        batch_adj = torch.stack([torch.Tensor(adj) for _ in range(h.shape[0])])
        batch_adj = batch_adj.to(device)

        hs = torch.sigmoid(self.conv1(h, batch_adj))
        hs = torch.sigmoid(self.conv2(hs, batch_adj))
        hs = torch.sigmoid(self.conv3(hs, batch_adj))
        hs = self.fc1(hs)
        p = torch.sigmoid(hs)
        p = p * (self.maxprob - self.minprob) + self.minprob

        # p = torch.clamp(p, min=0.0, max=1.0)
        #
        # if torch.isnan(p).any() or torch.isinf(p).any() or (p < 0).any() or (p > 1).any():
        #     print("Invalid p values detected")
        #     print(f"p min: {p.min().item()}, p max: {p.max().item()}")
        #     raise ValueError('Invalid p values detected')
        #
        # # if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
        # #     print("p 값에 NaN 또는 Inf 값이 포함되어 있습니다.")
        # #     print(f"p 값: {p}")
        # #     raise ValueError('p 값에 NaN 또는 Inf 값이 포함되어 있습니다.')

        u = torch.bernoulli(p).to(device)

        return u, p
class CondNet(nn.Module):
    def __init__(self):
        super.__init__()
        mlp_hidden = 1024
        output_dim = 10

        n_each_policylayer = 1
        # n_each_policylayer = 1 # if you have only 1 layer perceptron for policy net
        self.policy_net = nn.ModuleList()
        temp = nn.ModuleList()
        temp.append(nn.Linear(self.input_dim, mlp_hidden))
        temp.append(nn.Linear(mlp_hidden, mlp_hidden))
        self.policy_net.append(temp)

        for i in range(len(self.mlp)-2):
            temp = nn.ModuleList()
            for j in range(n_each_policylayer):
                temp.append(nn.Linear(self.mlp[i].out_features, self.mlp[i].out_features))
            self.policy_net.append(temp)
        self.policy_net.to(self.device)
class RuntimePruning(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_acc = L.Accuracy()
        self.val_acc = L.Accuracy()
        self.test_acc = L.Accuracy()
        self.train_loss = L.Loss()
        self.val_loss = L.Loss()
        self.test_loss = L.Loss()

def main():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=7)
    args.add_argument('--lambda_v', type=float, default=1.2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=40)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--learning_rate', type=float, default=1e-2)
    args.add_argument('--BATCH_SIZE', type=int, default=5)
    args.add_argument('--hidden-size', type=int, default=128)
    args.add_argument('--accum-step', type=int, default=20)
    args.add_argument('--allow_tf32', type=int, default=0)
    args.add_argument('--benchmark', type=int, default=0)
    args.add_argument('--precision', type=str, default='bf16')
    args.add_argument('--accelerator', type=str, default=device)
    args.add_argument('--matmul_precision', type=str, default='high')
    args.add_argument('--debug', type=bool, default=True)
    args = args.parse_args()

    if args.allow_tf32 == 1:
        args.allow_tf32 = True
    else:
        args.allow_tf32 = False

    if args.benchmark == 1:
        args.benchmark = True
    else:
        args.benchmark = False

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    if args.matmul_precision == 'medium':
        torch.set_float32_matmul_precision("medium")
    elif args.matmul_precision == 'high':
        torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="CONDG_RVS2", entity='hails', name='onlyrl_resnet50_imagenet',
                         config=args.__dict__)

    time = datetime.now()
    dir2save = '/Users/dongjaekim/Documents/imagenet'
    data_module = ImageNetDataModule(data_dir=dir2save, batch_size=args.BATCH_SIZE * args.accum_step, debug=args.debug)

    resnet = ResNet50(device)
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = False

    gnn_policy = Gnn(args.condnet_min_prob, args.condnet_max_prob, hidden_size=args.hidden_size, device=device).to(device)
    adj_, num_channels_ls = adj(resnet)

    model = runtime_pruner(resnet, gnn_policy, adj_, num_channels_ls, args, device)

    logger.watch(model, log_graph=False)

    # trainer = L.Trainer(
    #     max_epochs=args.max_epochs,
    #     accelerator='gpu',
    #     precision=args.precision,
    #     check_val_every_n_epoch=2,
    #     logger=logger
    # )
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        check_val_every_n_epoch=2,
        logger=logger
    )
    trainer.fit(model=model, datamodule=data_module)

    elapsed_time = datetime.now() - time
    print('Elapsed time: ', elapsed_time, 'minutes')
    wandb.log({'elapsed_time': elapsed_time.seconds})
    wandb.finish()

    print('')

if __name__ == '__main__':
    main()
