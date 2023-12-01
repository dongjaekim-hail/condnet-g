import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
from torch.nn.utils import prune

import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold

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
            weight_mean = torch.mean(self.mlp[i].weight.abs(), axis = 1)
            sorted_weight_mean, _ = torch.sort(weight_mean, descending=True)
            mask = sorted_weight_mean >= sorted_weight_mean[round(len(sorted_weight_mean) * self.tau)]
            h = F.relu(self.mlp[i](h))*mask
            layer_masks.append(mask.float())
        weight_mean = torch.mean(self.mlp[-1].weight.abs(), axis=1)
        sorted_weight_mean, _ = torch.sort(weight_mean, descending=True)
        mask = sorted_weight_mean >= sorted_weight_mean[round(len(sorted_weight_mean) * self.tau)]
        h = self.mlp[-1](h)*mask
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
    args.add_argument('--lambda_s', type=float, default=7)
    args.add_argument('--lambda_v', type=float, default=1.2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=40)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=256)
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

    wandb.init(project="condgnet",
                config=args.__dict__,
                name='magnitude_runtime' + '_tau=' + str(args.tau)
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

            # wandb log training/batch
            wandb.log({'train/batch_cost': loss.item(), 'train/batch_acc': acc, 'train/batch_tau': tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, loss.item(),
                                                                                         acc, np.mean([tau_.mean().item() for tau_ in layer_masks])))

            # wandb log training/epoch
            wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': tau})

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

                # wandb log test/epoch
                wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn,
                           'test/epoch_tau': taus / bn})
                # addup loss and acc
                costs += loss.to('cpu').item()
                accs += acc

                # tau_ = torch.stack(policies).mean().detach().item()
                # taus += tau_
            #print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log test/epoch
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn })
        torch.save(model.state_dict(), './cond_magnitude_weightMean_based_1024_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
    wandb.finish()
if __name__=='__main__':
    main()