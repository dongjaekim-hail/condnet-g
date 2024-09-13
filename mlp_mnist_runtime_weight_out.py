import torch
import torch.optim as optim
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import wandb
from datetime import datetime

wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")


class model_magnitude(nn.Module):
    def __init__(self,args):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.input_dim = 28*28
        mlp_hidden = [512, 256, 10]
        output_dim = mlp_hidden[-1]

        nlayers = args.nlayers

        self.mlp_nlayer = 0
        self.tau = args.tau

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.input_dim, mlp_hidden[0]))
        for i in range(nlayers):
            self.mlp.append(nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
        self.mlp.append(nn.Linear(mlp_hidden[i+1], output_dim))
        self.mlp.to(self.device)

    def forward(self, x):
        h = x.view(-1, self.input_dim).to(self.device)
        layer_masks = []
        for i in range(len(self.mlp) - 1):
            weight_magnitudes = torch.mean(self.mlp[i].weight.abs(), dim=1)

            sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)

            threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]

            mask = weight_magnitudes >= threshold

            h = F.relu(self.mlp[i](h)) * mask

            layer_masks.append(mask.float())

        weight_magnitudes = torch.mean(self.mlp[-1].weight.abs(), dim=1)

        sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)

        threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]

        mask = weight_magnitudes >= threshold

        h = self.mlp[-1](h)*mask

        layer_masks.append(mask.float())

        # softmax
        h = F.softmax(h, dim=1)
        layer_masks.append(mask.float())

        return h, layer_masks

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
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=30)
    args.add_argument('--condnet_min_prob', type=float, default=1e-3)
    args.add_argument('--condnet_max_prob', type=float, default=1 - 1e-3)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=200)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args = args.parse_args()
    lambda_l2 = args.lambda_l2
    tau = args.tau
    learning_rate = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE


    # datasets load mnist data
    train_dataset = datasets.MNIST(
        root="../data/mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.MNIST(
        root="../data/mnist",
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

    wandb.init(project="condgnet_edit5",
                config=args.__dict__,
                name='out_runtime_weight_magnitude' + '_tau=' + str(args.tau) + '_' + dt_string
                )

    # create model
    model = model_magnitude(args)
    # model = model_condnet2()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.mlp.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(train_loader, 0):
            mlp_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data

            outputs, layer_masks = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            loss = C(outputs, labels.to(model.device))
            loss.backward()
            mlp_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
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

                # make one hot vector
                y_batch_one_hot = torch.zeros(labels.shape[0], 10)
                y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1,).tolist()] = 1

                # get output
                outputs, layer_masks = model(torch.tensor(inputs))

                # calculate accuracy
                pred = torch.argmax(outputs, dim=1).to('cpu')
                acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                loss = C(outputs, labels.to(model.device))

                # addup loss and acc
                costs += loss.to('cpu').item()
                accs += acc
                taus += np.mean([tau_.mean().item() for tau_ in layer_masks])

            #print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log training/epoch
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn,
                       'test/epoch_tau': taus / bn})

        torch.save(model.state_dict(), './out_runtime_magnitude_weight' + '_tau=' + str(args.tau) + dt_string +'.pt')

    wandb.finish()
if __name__=='__main__':
    main()