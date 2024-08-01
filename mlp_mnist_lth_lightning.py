# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
import pickle
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from datasets import load_from_disk
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import wandb
from tqdm import trange

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_dataset = None
        self.val_dataset = None

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        return self.train_dataset[idx]
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
            self.val_dataset = datasets.MNIST('./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

# Function for Initialization
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
class mlp(nn.Module):
    def __init__(self, num_classes=10):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x)
        return x


# ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(
            f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(
        f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
    return (round((nonzero / total) * 100, 1))


# ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Prune by Percentile module
def prune_by_percentile(conv_percent, fc_percent, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    step = 0
    for name, param in model.named_parameters():

        if 'weight' in name:
            if "fc.weight" in name:
                percentile_value = np.percentile(abs(param.data.cpu().numpy()), fc_percent)
            else:
                percentile_value = np.percentile(abs(param.data.cpu().numpy()), conv_percent)

            tensor = param.data.cpu().numpy()
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0


# Function to make an empty mask of the same size as the model
def make_mask_local(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0
    return model, mask, step

def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0


class lth_trainer(L.LightningModule):
    def __init__(self, model, args, device, ite, resample = False, ITE=0, mask=None, step=None, best_accuracy = 0.0):
        super().__init__()
        self.model = model
        # Weight Initialization
        self.model.apply(weight_init)
        self.lr = args.lr
        self.criterion = nn.CrossEntropyLoss()
        self.automatic_optimization = False
        self.reinit = True if args.prune_type == "reinit" else False
        self.ite = ite
        self.args = args
        self._device = device
        self.resample = resample
        self.step = step
        self.mask = mask
        self.loss = []
        self.acc  = []
        self.best_accuracy = best_accuracy

        for name, param in model.named_parameters():
            print(name, param.size())

        self._prune_by_percentile(self.args.prune_percent_conv, self.args.prune_percent_fc)
        if self.reinit:
            self.model.apply(weight_init)
            step = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                    step = step + 1
            step = 0
        else:
            ValueError("Current version only supports reinit. Please use reinit=True.")
            # original_initialization(mask, initial_state_dict)

        print(f"\n--- Pruning Level [{ITE}:{self.ite}/{args.prune_iterations}]: ---")

    def _print_nonzeros(self):
        nonzero = total = 0
        for name, p in self.model.named_parameters():
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(
                f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        print(
            f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
        return (round((nonzero / total) * 100, 1))

    def _prune_by_percentile(self, conv_percent, fc_percent):
        mask = self.mask
        if self.ite > 0:
            step = 0
            for name, param in self.model.named_parameters():

                if 'weight' in name:
                    if "fc.weight" in name:
                        percentile_value = np.percentile(abs(param.data.cpu().numpy()), fc_percent)
                    else:
                        percentile_value = np.percentile(abs(param.data.cpu().numpy()), conv_percent)

                    tensor = param.data.cpu().numpy()
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
            self.mask = mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        EPS = 1e-6
        optimizer = self.optimizers()
        optimizer.zero_grad()
        # imgs, targets = next(train_loader)
        imgs, targets = batch
        output = self.model(imgs)
        train_loss = self.criterion(output, targets)
        train_loss.backward()
        self.loss.append(train_loss.item())

        # calculate accuracy
        pred = output.data.max(1, keepdim=True)[1]
        accs = pred.eq(targets.data.view_as(pred)).sum().item()
        accs = accs / imgs.size(0)
        self.acc.append(accs)

        # Freezing Pruned weights by making their gradients Zero
        for name, p in self.model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(self.device)

        optimizer.step()
        self.log('train/loss', train_loss.item(), prog_bar=True)
        self.log('train/accuracy', accs, prog_bar=True)
        # self.log(
        #     {'train/loss': train_loss.item(), 'train/accuracy': accs})

    def validation_step(self, batch, batch_idx):
        EPS = 1e-6
        # imgs, targets = next(test_loader)
        imgs, targets = batch
        output = self.model(imgs)
        val_loss = self.criterion(output, targets)

        # calculate accuracy
        pred = output.data.max(1, keepdim=True)[1]
        val_acc = pred.eq(targets.data.view_as(pred)).float().mean()

        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc.item()

        self.log('val/loss', val_loss.item(), prog_bar=True)
        self.log('val/accuracy', val_acc.item(), prog_bar=True)
        self.log('val/best_accuracy', self.best_accuracy, prog_bar=True)
        return val_loss, val_acc

# Main
def main(ITE=0):

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    time = datetime.now()
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20000, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=10, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_percent_conv", default=10, type=int, help="Pruning percent for conv layers")
    parser.add_argument("--prune_percent_fc", default=20, type=int, help="Pruning percent for fc layers")
    parser.add_argument("--prune_iterations", default=30, type=int, help="Pruning iterations count")
    args = parser.parse_args()

    wandb.init(project="condgnetre", entity='hails', name='mlp_mnist_lth', config=args.__dict__)
    wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    resample = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type == "reinit" else False

    logger = WandbLogger(project="lth_test", entity='hails', name='lth',
                         config=args.__dict__)

    datamodule = MNISTDataModule(batch_size=args.batch_size)

    # Importing Network Architecture
    # global model
    model = mlp().to(device)
    trainer = lth_trainer(model, args, device, -1, resample, ITE)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(trainer.model.state_dict())
    checkdir(f"{os.getcwd()}/saves/mlp/mnist/")
    torch.save(trainer.model,
               f"{os.getcwd()}/saves/mlp/mnist/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    model, mask, step = make_mask_local(model)

    # Optimizer and Loss
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss()  # Default was F.nll_loss

    # Layer Looper
    # for name, param in model.named_parameters():
    #     print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    step = 0
    all_loss = [] #np.zeros(args.end_iter, float)
    all_accuracy = [] #np.zeros(args.end_iter, float)
    comp = []

    for _ite in trange(args.start_iter, ITERATION):

        trainer = lth_trainer(model, args, device, _ite, resample, ITE, mask = mask, best_accuracy = best_accuracy)
        L_train = L.Trainer(
            max_epochs=args.end_iter,
            precision=args.precision,
            check_val_every_n_epoch=args.valid_freq,
            logger=logger
        )
        L_train.fit(trainer, datamodule)

        best_accuracy = trainer.best_accuracy
        mask = trainer.mask
        all_loss.extend(trainer.loss)
        all_accuracy.extend(trainer.acc)
        comp1 = trainer._print_nonzeros()
        comp.append(comp1)








if __name__ == "__main__":
    main(ITE=1)