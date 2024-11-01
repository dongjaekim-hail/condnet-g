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
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, debug: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            # 짧은 변을 [256, 480] 사이에서 무작위로 리사이즈하고, 224x224로 크롭
            transforms.RandomResizedCrop(224, scale=(256 / 480, 1.0)),
            # 무작위로 수평 뒤집기 적용
            transforms.RandomHorizontalFlip(),
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
        self.device = device

    def forward(self, x, sampling_size=(7, 7), cond_drop=False, us=None, channels=None):
        hs = [torch.flatten(F.interpolate(x, size=sampling_size), 2).to(self.device)]

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
            hs.append(torch.flatten(F.interpolate(x, size=sampling_size), 2).to(self.device))
            x = self.max_pool(x)
            count = 0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out)
                    hs.append(torch.flatten(F.interpolate(out, size=sampling_size), 2).to(self.device))
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    hs.append(torch.flatten(F.interpolate(out, size=sampling_size), 2).to(self.device))
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                        # hs.append(torch.flatten(F.interpolate(residual, size=sampling_size), 2).to(self.device))
                    out += residual
                    out = bottleneck.relu(out)
                    # hs.append(torch.flatten(F.interpolate(out, size=sampling_size), 2).to(self.device))
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
            us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1).to(self.device)
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
                    us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1).to(self.device)
                    out = out * us_
                    idx += 1
                    hs.append(out)
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].view(-1, layer_cumsum[idx + 2] - layer_cumsum[idx + 1], 1, 1).to(self.device)
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
            us_ = us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]].to(self.device)
            x = x * us_

        return x, hs

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
        self.args = args  # 여기에 args를 저장하세요
        self.automatic_optimization = False

        self.e_time_resnet_eval = []
        self.e_time_gnn_forward = []
        self.e_time_resnet_pruned_forward = []
        self.e_time_loss_backward = []
        self.e_time_resnet_opt_step = []
        self.e_time_gnn_opt_step = []

    def training_step(self, batch, batch_idx):
        resnet_opt, gnn_opt = self.optimizers()
        resnet_opt.zero_grad()
        gnn_opt.zero_grad()

        c_ = 0
        acc_ = 0
        acc_bf_ = 0
        PG_ = 0
        tau_ = 0
        L_ = 0
        Lb_total = 0
        Le_total = 0
        Lv_total = 0

        x, y = batch
        accum_batch_size = len(x) // self.accum_step
        for i in range(0, len(x), accum_batch_size):
            x_batch = x[i:i + accum_batch_size]
            y_batch = y[i:i + accum_batch_size]

            # # resnet surrogate forward
            # torch.cuda.synchronize()
            # start_time = time.time()
            self.resnet.eval()
            with torch.no_grad():
                y_hat_surrogate, hs = self.resnet(x_batch)
                acc_bf = accuracy(torch.argmax(y_hat_surrogate, dim=1), y_batch, task='multiclass', num_classes=1000)
            # torch.cuda.synchronize()
            # self.e_time_resnet_eval.append(time.time() - start_time)

            current_batch_size = hs[0].shape[0]

            if current_batch_size < self.args.BATCH_SIZE:
                adj_batch = self.adj[:current_batch_size]
            else:
                adj_batch = self.adj

            # # gnn forward
            # torch.cuda.synchronize()
            # start_time = time.time()
            us, p = self.gnn_policy(hs, adj_batch)
            # torch.cuda.synchronize()
            # self.e_time_gnn_forward.append(time.time() - start_time)

            # # print(f"p values: {p}")
            # # print(f"p min: {p.min()}, p max: {p.max()}")
            # #
            # # # Check for NaN or Inf values in p
            # # if torch.isnan(p).any() or torch.isinf(p).any():
            # #     print("NaN or Inf detected in p")
            # #     print(f"p: {p}")
            # #     raise ValueError('NaN or Inf detected in p')
            #
            # # Clipping p to ensure it's in the valid range [0, 1]
            # if torch.isnan(p).any() or torch.isinf(p).any() or (p < 0).any() or (p > 1).any():
            #     print("Invalid p values detected during training")
            #     print(f"p min: {p.min().item()}, p max: {p.max().item()}")
            #     raise ValueError('Invalid p values detected during training')
            # p = torch.clamp(p, min=0.0, max=1.0)
            # # print(f"p after clipping: {p}")
            # # print(f"p min after clipping: {p.min()}, p max after clipping: {p.max()}")

            # # resnet pruned forward
            # torch.cuda.synchronize()
            # start_time = time.time()
            self.resnet.train()
            outputs, hs = self.resnet(x_batch, sampling_size=(7, 7), cond_drop=True, us=us.detach(), channels=self.num_channels_ls)
            # torch.cuda.synchronize()
            # self.e_time_resnet_pruned_forward.append(time.time() - start_time)

            # # loss computation and backward
            # torch.cuda.synchronize()
            # start_time = time.time()
            c = F.cross_entropy(outputs, y_batch)
            # if torch.isnan(c) or torch.isinf(c):
            #     print("NaN or Inf in cross_entropy")
            #     print(f"outputs: {outputs}")
            #     print(f"y_batch: {y_batch}")
            #     raise ValueError('NaN or Inf detected in cross_entropy')
            # # c = F.cross_entropy(F.softmax(outputs, dim=1), y_batch)
            Lb_ = torch.pow(p.squeeze().mean(axis=0) - torch.tensor(self.tau).to(self.device), 2).mean()
            Le_ = torch.pow(p.squeeze().mean(axis=1) - torch.tensor(self.tau).to(self.device), 2).mean()

            L = c + self.lambda_s * (Lb_ + Le_)

            Lv_ = -torch.norm(p.squeeze() - p.squeeze().mean(axis=0), p=2, dim=0).mean()
            # Lv_ = (-1)* (p.squeeze().var(axis=0).mean()).mean()

            L += self.lambda_v * Lv_
            logp = torch.log(p.squeeze()).sum(axis=1).mean()
            # logp = torch.log(p.squeeze() + 1e-10).sum(axis=1).mean()  # Avoid log(0)
            PG = self.lambda_pg * c * (-logp) + L
            PG /= self.accum_step
            self.manual_backward(PG)
            # torch.cuda.synchronize()
            # self.e_time_loss_backward.append(time.time() - start_time)

            tau__ = us.mean().detach().item()

            c_ += c.item()
            acc_ += accuracy(torch.argmax(outputs, dim=1), y_batch, task='multiclass', num_classes=1000).item()
            acc_bf_ += acc_bf.item()
            PG_ += PG.item()
            L_ += L.item()
            tau_ += tau__

            # Accumulate Lb, Le, Lv values
            Lb_total += Lb_.item()
            Le_total += Le_.item()
            Lv_total += Lv_.item()

        # # resnet optimizer step
        # torch.cuda.synchronize()
        # start_time = time.time()
        resnet_opt.step()
        # torch.cuda.synchronize()
        # self.e_time_resnet_opt_step.append(time.time() - start_time)

        # # gnn optimizer step
        # torch.cuda.synchronize()
        # start_time = time.time()
        gnn_opt.step()
        # torch.cuda.synchronize()
        # self.e_time_gnn_opt_step.append(time.time() - start_time)

        # Log metrics
        c = c_ / self.accum_step
        acc = acc_ / self.accum_step
        acc_bf = acc_bf_ / self.accum_step
        PG = PG_ / self.accum_step
        L = L_ / self.accum_step
        tau = tau_ / self.accum_step
        Lb = Lb_total / self.accum_step
        Le = Le_total / self.accum_step
        Lv = Lv_total / self.accum_step

        self.log('train/batch_cost', c, prog_bar=True)
        self.log('train/batch_acc', acc, prog_bar=True)
        self.log('train/batch_acc_bf', acc_bf, prog_bar=True)
        self.log('train/batch_pg', PG, prog_bar=True)
        self.log('train/batch_loss', L, prog_bar=True)
        self.log('train/batch_tau', tau, prog_bar=True)

        # Log Lb, Le, Lv
        self.log('train/batch_Lb', Lb, prog_bar=True)
        self.log('train/batch_Le', Le, prog_bar=True)
        self.log('train/batch_Lv', Lv, prog_bar=True)

        # if any(len(times) >= 100 for times in [
        #     self.e_time_resnet_eval,
        #     self.e_time_gnn_forward,
        #     self.e_time_resnet_pruned_forward,
        #     self.e_time_loss_backward,
        #     self.e_time_resnet_opt_step,
        #     self.e_time_gnn_opt_step
        # ]):
        #     self.trainer.should_stop = True

    def validation_step(self, batch, batch_idx):
        x, y = batch

        c_ = 0
        acc_ = 0
        acc_bf_ = 0
        PG_ = 0
        L_ = 0
        tau_ = 0
        Lb_total = 0
        Le_total = 0
        Lv_total = 0

        accum_batch_size = len(x) // self.accum_step
        for i in range(0, len(x), accum_batch_size):
            x_batch = x[i:i + accum_batch_size]
            y_batch = y[i:i + accum_batch_size]

            self.resnet.eval()
            with torch.no_grad():
                y_hat_surrogate, hs = self.resnet(x_batch)
                acc_bf = accuracy(torch.argmax(y_hat_surrogate, dim=1), y_batch, task='multiclass', num_classes=1000)

                # current_batch_size와 adj_batch 동적으로 설정
                current_batch_size = hs[0].shape[0]
                if current_batch_size < self.args.BATCH_SIZE:
                    adj_batch = self.adj[:current_batch_size]
                else:
                    adj_batch = self.adj

                us, p = self.gnn_policy(hs, adj_batch)

                # if torch.isnan(p).any() or torch.isinf(p).any() or (p < 0).any() or (p > 1).any():
                #     print("Invalid p values detected during validation")
                #     print(f"p min: {p.min().item()}, p max: {p.max().item()}")
                #     raise ValueError('Invalid p values detected during validation')
                #
                #     # Clipping p to ensure it's in the valid range [0, 1]
                # p = torch.clamp(p, min=0.0, max=1.0)

                outputs, hs = self.resnet(x_batch, sampling_size=(7, 7), cond_drop=True, us=us.detach(), channels=self.num_channels_ls)
                # c = F.cross_entropy(F.softmax(outputs, dim=1), y_batch)
                c = F.cross_entropy(outputs, y_batch)
                # Lb_, Le_, Lv_ 계산 및 손실에 추가
                Lb_ = torch.pow(p.squeeze().mean(axis=0) - torch.tensor(self.tau).to(self.device), 2).mean()
                Le_ = torch.pow(p.squeeze().mean(axis=1) - torch.tensor(self.tau).to(self.device), 2).mean()
                L = c + self.lambda_s * (Lb_ + Le_)

                Lv_ = -torch.norm(p.squeeze() - p.squeeze().mean(axis=0), p=2, dim=0).mean()
                L += self.lambda_v * Lv_

                logp = torch.log(p.squeeze()).sum(axis=1).mean()
                # logp = torch.log(p.squeeze() + 1e-10).sum(axis=1).mean()  # Avoid log(0)
                PG = self.lambda_pg * c * (-logp) + L
                PG /= self.accum_step

                tau__ = us.mean().detach().item()

            # Update accumulated values
            c_ += c.item()
            acc_ += accuracy(torch.argmax(outputs, dim=1), y_batch, task='multiclass', num_classes=1000).item()
            acc_bf_ += acc_bf.item()
            PG_ += PG.item()
            L_ += L.item()
            tau_ += tau__
            Lb_total += Lb_.item()
            Le_total += Le_.item()
            Lv_total += Lv_.item()

        # Log metrics
        c = c_ / self.accum_step
        acc = acc_ / self.accum_step
        acc_bf = acc_bf_ / self.accum_step
        PG = PG_ / self.accum_step
        L = L_ / self.accum_step
        tau = tau_ / self.accum_step
        Lb = Lb_total / self.accum_step
        Le = Le_total / self.accum_step
        Lv = Lv_total / self.accum_step

        self.log('test/batch_cost', c, prog_bar=True)
        self.log('test/batch_acc', acc, prog_bar=True)
        self.log('test/batch_acc_bf', acc_bf, prog_bar=True)
        self.log('test/batch_pg', PG, prog_bar=True)
        self.log('test/batch_loss', L, prog_bar=True)
        self.log('test/batch_tau', tau, prog_bar=True)

        # Log Lb, Le, Lv
        self.log('test/batch_Lb', Lb, prog_bar=True)
        self.log('test/batch_Le', Le, prog_bar=True)
        self.log('test/batch_Lv', Lv, prog_bar=True)

        return {'val_loss': L}  # 전체 손실을 반환

    def configure_optimizers(self):
        resnet_optimizer = torch.optim.SGD(self.resnet.parameters(), lr=0.1,
                                           momentum=0.9, weight_decay=0.0001)
        resnet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            resnet_optimizer,
            mode='min',  # 손실이 최소일 때 적용
            factor=0.1,  # 학습률을 0.1배로 감소
            patience=10,  # 10번의 epoch 동안 개선되지 않으면 감소
            verbose=True  # 학습률 감소 시 출력
        )
        gnn_optimizer = torch.optim.SGD(self.gnn_policy.parameters(), lr=self.learning_rate,
                                        momentum=0.9, weight_decay=self.lambda_l2)
        return [resnet_optimizer, gnn_optimizer], [{'scheduler': resnet_scheduler, 'monitor': 'val_loss'}]

        # # GNN 옵티마이저 설정 (SGD, 초기 학습률 0.1, 모멘텀 0.9, 가중치 감쇠 0.0001)
        # gnn_optimizer = torch.optim.SGD(
        #     self.gnn_policy.parameters(),
        #     lr=0.1,  # 초기 학습률
        #     momentum=0.9,  # 모멘텀
        #     weight_decay=0.0001  # 가중치 감쇠
        # )
        #
        # # 손실이 감소하지 않을 때 학습률 감소 (오류가 평탄해질 때 10배 감소)
        # gnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     gnn_optimizer,
        #     mode='min',  # 손실이 최소일 때 적용
        #     factor=0.1,  # 학습률을 0.1배로 감소
        #     patience=10,  # 10번의 epoch 동안 개선되지 않으면 감소
        #     verbose=True  # 학습률 감소 시 출력
        # )
        #
        # # 옵티마이저와 스케줄러를 함께 반환
        # return [gnn_optimizer], [gnn_scheduler]

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            # Sanity Check 중에는 파일을 저장하지 않음
            return
        torch.save(self.resnet.state_dict(),
                   './resnet18_onlyrl_resnet_model_' + 's=' + str(self.lambda_s) + '_v=' + str(
                       self.lambda_v) + '_tau=' + str(
                       self.tau) + dt_string + '.pt')
        torch.save(self.gnn_policy.state_dict(),
                   './resnet18_onlyrl_gnn_policy_' + 's=' + str(self.lambda_s) + '_v=' + str(
                       self.lambda_v) + '_tau=' + str(
                       self.tau) + dt_string + '.pt')

class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, batch, conv_len, fc_len, adj_nodes, hidden_size = 64, device ='cpu'):
        super().__init__()
        self.conv1 = DenseSAGEConv(7*7, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.minprob = minprob
        self.maxprob = maxprob
        self.device = device
        self.conv_len = conv_len
        self.fc_len = fc_len

    def forward(self, hs, batch_adj):
        device = self.device
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
    import argparse
    args = argparse.ArgumentParser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=1.0)
    args.add_argument('--lambda_v', type=float, default=0.3)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.3)
    args.add_argument('--max_epochs', type=int, default=90)
    args.add_argument('--condnet_min_prob', type=float, default=0.01)
    args.add_argument('--condnet_max_prob', type=float, default=0.99)
    args.add_argument('--learning_rate', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=16)
    args.add_argument('--hidden-size', type=int, default=64)
    args.add_argument('--accum-step', type=int, default=16)
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

    logger = WandbLogger(project="condg_cnn", entity='hails', name='condg_resnet18_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau),
                         config=args.__dict__)

    time = datetime.now()
    # dir2save = r'C:\Users\97dnd\anaconda3\envs\torch\pr\resnet\data'
    dir2save = r'C:\Users\97dnd\anaconda3\envs\torch\pr\condnet-g\data'
    data_module = ImageNetDataModule(data_dir=dir2save, batch_size=args.BATCH_SIZE * args.accum_step, debug=args.debug)

    resnet = ResNet50(device)
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = True

    adj_, nodes_, conv_len, fc_len = adj(resnet)
    adj_ = torch.stack([torch.Tensor(adj_) for _ in range(args.BATCH_SIZE)]).to(device)

    gnn_policy = Gnn(args.condnet_min_prob, args.condnet_max_prob, batch=args.BATCH_SIZE,
                     conv_len=conv_len, fc_len=fc_len, adj_nodes=len(nodes_),
                     hidden_size=args.hidden_size).to(device)

    model = runtime_pruner(resnet, gnn_policy, adj_, nodes_, args, device)

    logger.watch(model, log_graph=False)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        precision=args.precision,
        check_val_every_n_epoch=1,
        logger=logger
    )
    trainer.fit(model=model, datamodule=data_module)

    elapsed_time = datetime.now() - time
    print('Elapsed time: ', elapsed_time, 'minutes')
    wandb.finish()


if __name__ == '__main__':
    main()
