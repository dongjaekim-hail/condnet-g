import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import DenseSAGEConv
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import wandb
from transformers import Trainer, EarlyStoppingCallback, TrainingArguments
from pynvml import *
from datasets import load_from_disk
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import WandbLogger

class CondNetModule(L.LightningModule):
    def __init__(self, args, resnet_model, gnn_policy, adj_, num_channels_ls):
        super().__init__()
        self.args = args
        self.resnet_model = resnet_model
        self.gnn_policy = gnn_policy
        self.adj_ = adj_
        self.num_channels_ls = num_channels_ls
        self.criterion = torch.nn.CrossEntropyLoss()

        # PyTorch Lightning에서는 모델 또는 옵티마이저에 대한 모든 설정을 `self`에 저장해야 합니다.
        self.save_hyperparameters()

    def forward(self, inputs, cond_drop=False, us=None, channels=None):
        # 모델의 순전파 로직을 여기에 정의합니다.
        return self.resnet_model(inputs, cond_drop, us, channels)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs_surrogate, hs = self.resnet_model(inputs)
        us, p = self.gnn_policy(hs, self.adj_)
        outputs, hs = self.forward(inputs, cond_drop=True, us=us.detach(), channels=self.num_channels_ls)

        loss = self.criterion(outputs, labels)
        # 손실, 로그 및 기타 값의 반환은 여기서 처리합니다.
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # 옵티마이저와 스케줄러
        resnet_optimizer = torch.optim.SGD(self.resnet_model.parameters(), lr=self.args.learning_rate, momentum=0.9,
                                           weight_decay=self.args.lambda_l2)
        policy_optimizer = torch.optim.SGD(self.gnn_policy.parameters(), lr=self.args.learning_rate, momentum=0.9,
                                           weight_decay=self.args.lambda_l2)
        return [resnet_optimizer, policy_optimizer], []

class ResNetModule(L.LightningModule):
    def __init__(self, args, resnet_model):
        super().__init__()
        self.args = args
        self.resnet_model = resnet_model
        # self.automatic_optimization = False
        # self.save_hyperparameters()

    def forward(self, inputs):
        return self.resnet_model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, _ = self.forward(inputs)
        preds = F.softmax(outputs, dim=1)
        # loss = self.criterion(preds, labels)
        loss = F.cross_entropy(preds, labels)
        train_acc = accuracy(torch.argmax(preds, dim=1), labels, task='multiclass', num_classes=1000)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, _ = self.forward(inputs)
        preds = F.softmax(outputs, dim=1)
        loss = F.cross_entropy(preds, labels)
        val_acc = accuracy(torch.argmax(preds, dim=1), labels, task='multiclass', num_classes=1000)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # 옵티마이저와 스케줄러
        resnet_optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate,
                                           weight_decay=self.args.lambda_l2)
        return resnet_optimizer

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
class Gnn(nn.Module):
    def __init__(self, minprob, maxprob, hidden_size = 64):
        super().__init__()
        self.conv1 = DenseSAGEConv(7*7, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(64,1, bias=False)
        self.minprob = minprob
        self.maxprob = maxprob

    def forward(self, hs, adj):
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
class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, debug: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transform = transform
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
        if self.debug:
            self.train_dataset = ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)
        else:
            self.train_dataset = ImagenetDataset(os.path.join(self.data_dir, 'train'), transform=self.transform)
        self.val_dataset =  ImagenetDataset(os.path.join(self.data_dir, 'validation'), transform=self.transform)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

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
    args.add_argument('--max_epochs', type=int, default=50)
    # args.add_argument('--condnet_min_prob', type=float, default=0.01)
    # args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--learning_rate', type=float, default=1e-2)
    args.add_argument('--BATCH_SIZE', type=int, default=100)
    # args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=256)
    args.add_argument('--accum-step', type=int, default=1)
    # parameters related to pytorch_lightning
    # args.add_argument('--allow_tf32', type=bool, default=True)
    args.add_argument('--allow_tf32', type=int, default=0)
    # args.add_argument('--benchmark', type=bool, default=True)
    args.add_argument('--benchmark', type=int, default=0)
    args.add_argument('--precision', type=str, default='32') # 'bf16', '32'
    args.add_argument('--accelerator', type=str, default=device)
    args.add_argument('--matmul_precision', type=str, default='high')
    args.add_argument('--debug', type=bool, default=True)
    args = args.parse_args()

    # if args.allow_tf32 == 1:
    #     args.allow_tf32 = True
    # else:
    #     args.allow_tf32 = False
    #
    # if args.benchmark == 1:
    #     args.benchmark = True
    # else:
    #     args.benchmark = False
    #
    # # device = 'cpu'
    # if args.allow_tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    # else:
    #     torch.backends.cuda.matmul.allow_tf32 = False
    # if args.benchmark:
    #     torch.backends.cudnn.benchmark = True
    # else:
    #     torch.backends.cudnn.benchmark = False
    # # if args.precision == 'bf16':
    # #     torch.set_default_dtype(torch.bfloat16)
    # # elif args.precision == '32':
    # #     torch.set_default_dtype(torch.float32)
    # # elif args.precision == '16-mixed':
    # #     torch.set_default_dtype(torch.float16)
    # # else:
    # #     raise ValueError('Invalid precision')
    # if args.matmul_precision == 'medium':
    #     torch.set_float32_matmul_precision("medium") # high
    # elif args.matmul_precision == 'high':
    #     torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="CONDG_RVS", entity='hails', name='resnet50_imagenet',
                            config=args.__dict__)
                         # config=args.__dict__, log_model='all')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=100,
        mode='min',
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)

    time = datetime.now()
    dir2save = '/Users/dongjaekim/Documents/imagenet'
    dir2save = 'D:/imagenet-1k/'
    train_dataset = ImageNetDataModule(dir2save, batch_size=args.BATCH_SIZE, debug=args.debug)

    elapsed_time = datetime.now() - time
    print('Data loading time: ', elapsed_time,'minutes')

    resnet = ResNet50(device).to(device)
    for param in resnet.parameters():
        param.requires_grad = True
    model = ResNetModule(args, resnet)

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='gpu',
        precision= args.precision,
        check_val_every_n_epoch=1,
        logger=logger,
        accumulate_grad_batches=args.accum_step,
    )

    #torch sync
    time= datetime.now()

    # logger.watch(model, log='all')
    logger.watch(model, log_graph=False)


    trainer.fit(model, train_dataset)

    model.to(device)
    optimizer = model.configure_optimizers()
    train_dataset.setup()
    trainloader =  train_dataset.train_dataloader()
    for i,  data in enumerate(trainloader):
        loss = model.training_step([data[0].cuda(), data[1].cuda()], 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break


    # print elpased time
    elapsed_time = datetime.now() - time
    print('Elapsed time: ', elapsed_time, 'minutes')
    wandb.log({'elapsed_time': elapsed_time.seconds})
    wandb.finish()

if __name__=='__main__':
    main()