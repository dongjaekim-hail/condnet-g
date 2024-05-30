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
import tqdm
import numpy as np
from torch_geometric.nn import DenseSAGEConv

class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, debug: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    # def setup(self, stage: str):
    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "fit":
    #         mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         self.mnist_train, self.mnist_val = random_split(
    #             mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
    #         )
    #
    #     # Assign test dataset for use in dataloader(s)
    #     if stage == "test":
    #         self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    #
    #     if stage == "predict":
    #         self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
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
            self.val_dataset =  ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)
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
        resnet_model = models.resnet50(pretrained=False)
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
            # for layer in [self.conv1, self.bn1, self.relu, self.max_pool]:
            #     x = layer(x)
            #     if 'ReLU' in str(layer):
            #         hs.append(torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(device))
            count=0
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
                count+=1
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
            # for layer in [self.conv1, self.bn1, self.relu, self.max_pool]:
                # x = layer(x)
                # if 'Conv' in str():
                #     x = x * us[:, channels[0]: channels[0] + channels[1]]
                # elif 'ReLU' in str(layer):
                #     hs.append(torch.flatten(F.interpolate(x, size=(7, 7)), 2).to(device))

            i = 0
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for bottleneck in layer:
                    # i = 0
                    # l_idx = []
                    # i_idx = []
                    # for l in bottleneck.children():
                    #     l_idx.append(l)
                    #     i_idx.append(i)
                    #     print(channels[i + 2])
                    #     i+=1


                    residual = x
                    out = bottleneck.conv1(x)
                    out = bottleneck.bn1(out)
                    out = bottleneck.relu(out) # 64

                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1

                    hs.append(out)
                    out = bottleneck.conv2(out)
                    out = bottleneck.bn2(out)
                    out = bottleneck.relu(out) # 64

                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1

                    hs.append(out)
                    out = bottleneck.conv3(out)
                    out = bottleneck.bn3(out)
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(x)
                    out += residual
                    out = bottleneck.relu(out) # 256

                    us_ = us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    out = out * us_
                    i += 1

                    hs.append(out)
                    x = out

                    # i = 0
                    # for l in bottleneck.children():
                    #     if 'Conv' in str(l):
                    #         out = out * us[:, channels[i + 1]:channels[i + 1] + channels[i + 2]]
                    #         i += 1

            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x, hs
def adj(resnet_model, bidirect = True, last_layer = True, edge2itself = True):
    # resnet_model = models.resnet50(pretrained=True)
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
                    # for l in bottleneck.children():
                    #     if isinstance(l, torch.nn.modules.conv.Conv2d):
                    #         num_channels_ls.append(l.in_channels)
                    #     elif isinstance(l, torch.nn.Sequential):
                    #         for sub_layer in l.children():
                    #             if isinstance(sub_layer, torch.nn.modules.conv.Conv2d):
                    #                 num_channels_ls.append(sub_layer.in_channels)
                #     for l in bottleneck.children():
                #         if 'Conv' in str(l):
                #             num_channels_ls.append(l.in_channels)
                # except Exception:
                #     continue
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
        # layer = model.layers[i]
        # num_current = layer.in_features
        # num_next = layer.out_features
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
        # make sure every element that is non-zero is 1
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

    def training_step(self, batch, batch_idx):
        resnet_opt, gnn_opt = self.optimizers()
        resnet_opt.zero_grad()
        gnn_opt.zero_grad()

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

            self.resnet.eval()
            with torch.no_grad():
                y_hat_surrogate, hs = self.resnet(x_batch)
                acc_bf = accuracy(torch.argmax(y_hat_surrogate, dim=1), y_batch, task='multiclass', num_classes=1000)
            us, p = self.gnn_policy(hs, self.adj)
            self.resnet.train()

            outputs, hs = self.resnet(x_batch, cond_drop=True, us=us.detach(), channels=self.num_channels_ls)
            c = F.cross_entropy(outputs, y_batch)
            acc = accuracy(torch.argmax(outputs, dim=1), y_batch, task='multiclass', num_classes=1000)
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

        resnet_opt.step()
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

                outputs, hs = self.resnet(x_batch, cond_drop=True, us=us.detach(), channels=self.num_channels_ls)
                c = F.cross_entropy(outputs, y_batch)
                acc = accuracy(torch.argmax(outputs, dim=1), y_batch, task='multiclass', num_classes=1000)
                L = c + self.lambda_s * (torch.pow(p.squeeze().mean(axis=0) - torch.tensor(self.tau), 2).mean() +
                                            torch.pow(p.squeeze().mean(axis=1) - torch.tensor(self.tau), 2).mean())
                L += self.lambda_v * (-1) * (p.squeeze().var(axis=0).mean() +
                                            p.squeeze().var(axis=1).mean())
                logp = torch.log(p.squeeze()).sum(axis=1).mean()
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
        resnet_optimizer = torch.optim.SGD(self.resnet.parameters(), lr=self.learning_rate,
                                  momentum=0.9, weight_decay=self.lambda_l2)
        gnn_optimizer = torch.optim.SGD(self.gnn_policy.parameters(), lr=self.learning_rate,
                                    momentum=0.9, weight_decay=self.lambda_l2)
        return resnet_optimizer, gnn_optimizer


class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, hidden_size = 64, device ='cpu'):
        super().__init__()
        self.conv1 = DenseSAGEConv(7*7, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(64,1, bias=False)
        self.minprob = minprob
        self.maxprob = maxprob
        self.device = device

    def forward(self, hs, adj):
        device = self.device
        # hs : hidden activity
        h = hs[0]
        for i in hs[1:]:
            h = torch.cat((h, i), dim=1)
        h = h.to(device)

        batch_adj = torch.stack([torch.Tensor(adj) for _ in range(h.shape[0])])
        batch_adj = batch_adj.to(device)

        # hs_0 = hs.unsqueeze(-1)

        hs = F.sigmoid(self.conv1(h, batch_adj))
        hs = F.sigmoid(self.conv2(hs, batch_adj))
        hs = F.sigmoid(self.conv3(hs, batch_adj))
        hs = self.fc1(hs)
        p = F.sigmoid(hs)
        # bernoulli sampling
        p = p * (self.maxprob - self.minprob) + self.minprob
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
    # get args
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
    args.add_argument('--max_epochs', type=int, default=10)
    # args.add_argument('--condnet_min_prob', type=float, default=0.01)
    # args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--learning_rate', type=float, default=1e-2)
    args.add_argument('--BATCH_SIZE', type=int, default=10)
    # args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args.add_argument('--accum-step', type=int, default=10)
    # parameters related to pytorch_lightning
    # args.add_argument('--allow_tf32', type=bool, default=True)
    args.add_argument('--allow_tf32', type=int, default=1)
    # args.add_argument('--benchmark', type=bool, default=True)
    args.add_argument('--benchmark', type=int, default=0)
    args.add_argument('--precision', type=str, default='16-true') # 'bf16':3090 or newer (ampere), '32', '16-true', '16-mixed'
    args.add_argument('--accelerator', type=str, default=device)
    args.add_argument('--matmul_precision', type=str, default='high')
    args.add_argument('--debug', type=bool, default=False)
    args = args.parse_args()

    if args.allow_tf32 == 1:
        args.allow_tf32 = True
    else:
        args.allow_tf32 = False

    if args.benchmark == 1:
        args.benchmark = True
    else:
        args.benchmark = False

    # device = 'cpu'
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    # if args.precision == 'bf16':
    #     torch.set_default_dtype(torch.bfloat16)
    # elif args.precision == '32':
    #     torch.set_default_dtype(torch.float32)
    # elif args.precision == '16':
    #     torch.set_default_dtype(torch.float16)
    # else:
    #     raise ValueError('Invalid precision')
    if args.matmul_precision == 'medium':
        torch.set_float32_matmul_precision("medium") # high
    elif args.matmul_precision == 'high':
        torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="CONDG_RVS2", entity='hails', name='resnet50_imagenet',
                            config=args.__dict__)
                         # config=args.__dict__, log_model='all')

    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     filename='resnet-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=100,
    #     mode='min',
    # )
    # early_stopping = EarlyStopping(monitor='val_loss', patience=100)

    time = datetime.now()

    dir2save = 'D:/imagenet-1k/'
    # dir2save = '/Users/dongjaekim/Documents/imagenet'

    data_module = ImageNetDataModule(data_dir=dir2save, batch_size=args.BATCH_SIZE*args.accum_step, debug=True)

    resnet = ResNet50(device)
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = True

    gnn_policy = Gnn(args.condnet_min_prob, args.condnet_max_prob, hidden_size=args.hidden_size, device=device).to(device)
    adj_, num_channels_ls = adj(resnet)

    model = runtime_pruner(resnet, gnn_policy, adj_, num_channels_ls, args, device)

    logger.watch(model, log_graph=False)
    # train model
    # trainer = L.Trainer()
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        precision= args.precision,
        check_val_every_n_epoch=2,
        logger=logger
    )
    trainer.fit(model=model, datamodule=data_module)

    # print elpased time
    elapsed_time = datetime.now() - time
    print('Elapsed time: ', elapsed_time, 'minutes')
    wandb.log({'elapsed_time': elapsed_time.seconds})
    wandb.finish()

    print('')


if __name__=='__main__':
    main()