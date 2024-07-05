import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
import pickle
from datetime import datetime
from tqdm import tqdm

sns.set_style('darkgrid')


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=self.transform)
        self.val_dataset = datasets.CIFAR10('../data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


# Check if directory exists and create if not
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Count non-zero parameters in model
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

# Weight initialization
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


class SimpleCNN(L.LightningModule):
    def __init__(self, lr, prune_percent_conv, prune_percent_fc, prune_type, initial_state_dict, prune_iterations, total_iterations):
        super(SimpleCNN, self).__init__()
        self.lr = lr
        self.prune_percent_conv = prune_percent_conv
        self.prune_percent_fc = prune_percent_fc
        self.prune_type = prune_type
        self.initial_state_dict = initial_state_dict
        self.prune_iterations = prune_iterations

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

        self.criterion = nn.CrossEntropyLoss()
        self.mask = None
        self.step = 0
        self.iteration = 0
        self.initial_state_dict = None
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.best_accuracy = 0.0
        self.validation_outputs = []

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        output = self(imgs)
        loss = self.criterion(output, targets)

        acc = (output.argmax(dim=1) == targets).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        # Print loss and accuracy to console
        print(f'Train Loss: {loss.item():.4f}, Train Accuracy: {acc.item():.4f}')

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        output = self(imgs)
        loss = self.criterion(output, targets)
        acc = (output.argmax(dim=1) == targets).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        # Print loss and accuracy to console
        self.validation_outputs.append({'val_loss': loss, 'val_acc': acc})

        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in self.validation_outputs]).mean()

        self.log('avg_val_loss', avg_val_loss)
        self.log('avg_val_acc', avg_val_acc)

        if avg_val_acc > self.best_accuracy:
            self.best_accuracy = avg_val_acc
            checkdir(f"{os.getcwd()}/saves/cnn/cifar10/")
            torch.save(self.state_dict(), f"{os.getcwd()}/saves/cnn/cifar10/best_model.pth.tar")

        tqdm.write(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}, Best Accuracy: {self.best_accuracy:.4f}')

        self.validation_outputs.clear()

    # def on_train_epoch_end(self):
    #     if self.iteration < self.prune_iterations:
    #         if self.iteration > 0:
    #             self.prune_by_percentile(self.prune_percent_conv, self.prune_percent_fc,
    #                                      reinit=(self.prune_type == "reinit"))
    #             if self.prune_type == "reinit":
    #                 self.apply(weight_init)
    #                 self.step = 0
    #                 for name, param in self.named_parameters():
    #                     if 'weight' in name:
    #                         weight_dev = param.device
    #                         param.data = torch.from_numpy(param.data.cpu().numpy() * self.mask[self.step]).to(
    #                             weight_dev)
    #                         self.step += 1
    #                 self.step = 0
    #             else:
    #                 self.original_initialization(self.mask, self.initial_state_dict)
    #             self.configure_optimizers()
    #         print(f"\n--- Pruning Level [{self.iteration}/{self.prune_iterations}]: ---")
    #         compression_rate = self.print_nonzeros()
    #         self.iteration += 1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.current_iteration += 1
        if self.current_iteration % self.total_iterations == 0 and self.current_iteration != 0 and self.current_iteration != self.total_iterations * self.prune_iterations:
            print(f"Pruning at iteration {self.current_iteration} (Batch {batch_idx})")
            self.prune_and_reset()

    def prune_and_reset(self):
        self.prune_by_percentile(self.prune_percent_conv, self.prune_percent_fc, reinit=(self.prune_type == "reinit"))
        if self.prune_type == "reinit":
            self.apply(self.weight_init)
            self.step = 0
            for name, param in self.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.cpu().numpy() * self.mask[self.step]).to(weight_dev)
                    self.step += 1
            self.step = 0
        else:
            self.original_initialization(self.mask, self.initial_state_dict)
        self.configure_optimizers()
        print(f"\n--- Pruning Level [{self.current_iteration // self.total_iterations}/{self.prune_iterations}]: ---")
        self.print_nonzeros()

    def make_mask(self):
        self.step = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                self.step += 1
        self.mask = [None] * self.step
        self.step = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                self.mask[self.step] = np.ones_like(tensor)
                self.step += 1
        self.step = 0

    def original_initialization(self, mask_temp, initial_state_dict):
        step = 0
        for name, param in self.named_parameters():
            if "weight" in name:
                weight_dev = param.device
                param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
                step += 1
            if "bias" in name:
                param.data = initial_state_dict[name].to(param.device)
        self.step = 0

    def print_nonzeros(self):
        nonzero = total = 0
        for name, p in self.named_parameters():
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

    def prune_by_percentile(self, conv_percent, fc_percent, resample=False, reinit=False, **kwargs):
        self.step = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                if "fc.weight" in name:
                    percentile_value = np.percentile(abs(param.data.cpu().numpy()), fc_percent)
                else:
                    percentile_value = np.percentile(abs(param.data.cpu().numpy()), conv_percent)

                tensor = param.data.cpu().numpy()
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, self.mask[self.step])

                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                self.mask[self.step] = new_mask
                self.step += 1
        self.step = 0


def main():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20000, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_percent_conv", default=10, type=int, help="Pruning percent for conv layers")
    parser.add_argument("--prune_percent_fc", default=20, type=int, help="Pruning percent for fc layers")
    parser.add_argument("--prune_iterations", default=30, type=int, help="Pruning iterations count")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    initial_state_dict = None

    model = SimpleCNN(
        lr=args.lr,
        prune_percent_conv=args.prune_percent_conv,
        prune_percent_fc=args.prune_percent_fc,
        prune_type=args.prune_type,
        initial_state_dict=initial_state_dict,
        prune_iterations=args.prune_iterations, total_iterations=args.end_iter
    )

    model.make_mask()

    model.initial_state_dict = copy.deepcopy(model.state_dict())

    datamodule = CIFAR10DataModule(batch_size=args.batch_size)
    datamodule.setup()

    trainer = L.Trainer(max_steps=args.end_iter * args.prune_iterations, devices=1 if torch.cuda.is_available() else 0,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, datamodule)

    # trainer = L.Trainer(max_epochs=args.end_iter, devices=1 if torch.cuda.is_available() else 0,
    #                      accelerator="gpu" if torch.cuda.is_available() else "cpu")
    # trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
