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

    def forward(self, x):
        layer_masks = []

        # Conv1
        weight_magnitudes = torch.mean(self.conv1.weight.abs(), dim=(1, 2, 3))
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        mask = mask.repeat(x.size(0), 1).view(x.size(0), -1, 1, 1)
        x = F.relu(self.conv1(x)) * mask
        layer_masks.append(mask.float())

        # Conv2
        weight_magnitudes = torch.mean(self.conv2.weight.abs(), dim=(1, 2, 3))
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        mask = mask.repeat(x.size(0), 1).view(x.size(0), -1, 1, 1)
        x = F.relu(self.conv2(x)) * mask
        x = self.pool(x)
        layer_masks.append(mask.float())

        # Conv3
        weight_magnitudes = torch.mean(self.conv3.weight.abs(), dim=(1, 2, 3))
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        mask = mask.repeat(x.size(0), 1).view(x.size(0), -1, 1, 1)
        x = F.relu(self.conv3(x)) * mask
        layer_masks.append(mask.float())

        # Conv4
        weight_magnitudes = torch.mean(self.conv4.weight.abs(), dim=(1, 2, 3))
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        mask = mask.repeat(x.size(0), 1).view(x.size(0), -1, 1, 1)
        x = F.relu(self.conv4(x)) * mask
        x = self.pool(x)
        layer_masks.append(mask.float())

        x = x.view(-1, 128 * 8 * 8)

        # FC1
        weight_magnitudes = torch.mean(self.fc1.weight.abs(), dim=1)
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        x = F.relu(self.fc1(x)) * mask
        x = self.dropout(x)
        layer_masks.append(mask.float())

        # FC2
        weight_magnitudes = torch.mean(self.fc2.weight.abs(), dim=1)
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        x = F.relu(self.fc2(x)) * mask
        x = self.dropout(x)
        layer_masks.append(mask.float())

        # FC3
        weight_magnitudes = torch.mean(self.fc3.weight.abs(), dim=1)
        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
        mask = weight_magnitudes >= threshold
        x = self.fc3(x) * mask
        layer_masks.append(mask.float())

        return x, layer_masks

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

    wandb.init(project="cond_cnn_cifar10_edit",
                config=args.__dict__,
                name='out_cnn_runtime_weight_magnitude' + '_tau=' + str(args.tau) + '_' + dt_string
                )

    # create model
    model = SimpleCNN(args)

    if torch.cuda.is_available():
        model = model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    C = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, layer_masks = model(inputs)

            loss = C(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += loss.to('cpu').item()
            accs += acc
            tau = np.mean([layer_mask.mean().item() for layer_mask in layer_masks])

            # wandb log training/batch
            wandb.log({'train/batch_cost': loss.item(), 'train/batch_acc': acc, 'train/batch_tau': tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, loss.item(),
                                                                                         acc, tau))

        # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': tau / bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0

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

                inputs, labels = inputs.to(device), labels.to(device)

                # get output
                outputs, layer_masks = model(torch.tensor(inputs))

                # calculate accuracy
                pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                loss = C(outputs, labels.to(device))

                # addup loss and acc
                costs += loss.to('cpu').item()
                accs += acc
                taus += np.mean([tau_.mean().item() for tau_ in layer_masks])

            # print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log training/epoch
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn,
                       'test/epoch_tau': taus / bn})

        torch.save(model.state_dict(), './out_cnn_runtime_magnitude_weight' + '_tau=' + str(args.tau) + dt_string + '.pt')

    wandb.finish()
if __name__ == '__main__':
    main()