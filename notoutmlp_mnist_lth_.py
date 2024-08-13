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
import torch.optim as optim

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

import torch
import torch.nn as nn

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

# Main
def main(ITE=0):

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    time = datetime.now()
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=151, type=int)
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

    wandb.init(project="condgnet_edit4", entity='hails', name='adam_mlp_mnist_lth', config=args.__dict__)
    wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    resample = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type == "reinit" else False

    # # Data Loader
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    # testdataset = datasets.MNIST('../data', train=False, transform=transform)
    # train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    #                                            drop_last=False)
    # # train_loader = cycle(train_loader)
    # test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    #                                           drop_last=True)

    transform = transforms.ToTensor()
    traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    testdataset = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                              drop_last=True)

    # Importing Network Architecture
    global model
    model = mlp().to(device)

    # Weight Initialization
    model.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    checkdir(f"{os.getcwd()}/saves/mlp/mnist/")
    torch.save(model,
               f"{os.getcwd()}/saves/mlp/mnist/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.1,
    #                       momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()  # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)

    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            prune_by_percentile(args.prune_percent_conv, args.prune_percent_fc, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model)
        comp[_ite] = comp1
        # pbar = tqdm(range(args.end_iter))
        pbar = tqdm(range(args.end_iter), dynamic_ncols=False)

        for iter_ in pbar:
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                test_accuracy = test(model, test_loader, criterion)

                # Save Weights
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    checkdir(f"{os.getcwd()}/saves/mlp/mnist/")
                    torch.save(model, f"{os.getcwd()}/saves/mlp/mnist/{_ite}_model_{args.prune_type}.pth.tar")

            # Training
            train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = train_loss
            all_accuracy[iter_] = test_accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.write(
                    f'Train Epoch: {iter_}/{args.end_iter} Train Loss: {train_loss:.6f} Train Accuracy: {train_accuracy:.2f}% Test Accuracy: {test_accuracy:.2f}% Best Test Accuracy: {best_accuracy:.2f}%')
                wandb.log(
                    {'train/epoch_L': train_loss, 'train/epoch_acc': train_accuracy, 'test/epoch_acc': test_accuracy,
                     'Best Test Accuracy': best_accuracy})
                wandb.log(
                    {f'Pruning Iteration {_ite}/train/epoch_L': train_loss,
                     f'Pruning Iteration {_ite}/train/epoch_acc': train_accuracy,
                     f'Pruning Iteration {_ite}/test/epoch_acc': test_accuracy,
                     f'Pruning Iteration {_ite}/Best Test Accuracy': best_accuracy})

        torch.save(model.state_dict(), f"{os.getcwd()}/saves/mlp/mnist/{_ite}_model_{args.prune_type}_final.pth.tar")
        writer.add_scalar('Accuracy/test', best_accuracy, comp1)
        bestacc[_ite] = best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        # NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1, (args.end_iter) + 1),
                 100 * (all_loss - np.min(all_loss)) / np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1, (args.end_iter) + 1), all_accuracy, c="red", label="Accuracy")
        plt.title(f"Loss Vs Accuracy Vs Iterations (mnist,mlp)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and Accuracy")
        plt.legend()
        plt.grid(color="gray")
        checkdir(f"{os.getcwd()}/plots/lt/mlp/mnist/")
        plt.savefig(
            f"{os.getcwd()}/plots/lt/mlp/mnist/{args.prune_type}_LossVsAccuracy_{comp1}.png",
            dpi=1200)
        plt.close()

        # Dump Plot values
        checkdir(f"{os.getcwd()}/dumps/lt/mlp/mnist/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/mlp/mnist/{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(
            f"{os.getcwd()}/dumps/lt/mlp/mnist/{args.prune_type}_all_accuracy_{comp1}.dat")

        # Dumping mask
        checkdir(f"{os.getcwd()}/dumps/lt/mlp/mnist/")
        with open(f"{os.getcwd()}/dumps/lt/mlp/mnist/{args.prune_type}_mask_{comp1}.pkl",
                  'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.end_iter, float)
        all_accuracy = np.zeros(args.end_iter, float)

    # Dumping Values for Plotting
    checkdir(f"{os.getcwd()}/dumps/lt/mlp/mnist/")
    comp.dump(f"{os.getcwd()}/dumps/lt/mlp/mnist/{args.prune_type}_compression.dat")
    bestacc.dump(f"{os.getcwd()}/dumps/lt/mlp/mnist/{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets")
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage (mnist,mlp)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test accuracy")
    plt.xticks(a, comp, rotation="vertical")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color="gray")
    checkdir(f"{os.getcwd()}/plots/lt/mlp/mnist/")
    plt.savefig(f"{os.getcwd()}/plots/lt/mlp/mnist/{args.prune_type}_AccuracyVsWeights.png",
                dpi=1200)
    plt.close()

    elapsed_time = datetime.now() - time
    print('Elapsed time: ', elapsed_time, 'minutes')
    wandb.log({'elapsed_time': elapsed_time.seconds})
    wandb.finish()


# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()

        epoch_loss += train_loss.item()

        # Calculate accuracy
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum().item()
        total += targets.size(0)

    epoch_loss /= len(train_loader)
    accuracy = correct / total
    return epoch_loss, accuracy


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
    return accuracy


# Prune by Percentile module
def prune_by_percentile(conv_percent, fc_percent, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    step = 0
    for name, param in model.named_parameters():

        if 'weight' in name:
            if "fc3.weight" in name:
                continue
            elif "fc.weight" in name:
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

if __name__ == "__main__":
    main(ITE=1)