import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.nn.utils import prune

import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime

class magnitude(nn.Module):
    def __init__(self, args):
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

        for layer in self.mlp:
            prune.custom_from_mask(layer, name="weight", mask=torch.ones_like(layer.weight))

    def forward(self, x):
        h = x.view(-1, self.input_dim).to(self.device)
        layer_masks = []

        for i in range(len(self.mlp) - 1):
            prune.custom_from_mask(self.mlp[i], name="weight", mask=torch.ones_like(self.mlp[i].weight))

            weight = self.mlp[i].weight

            threshold = torch.topk(torch.abs(weight).mean(dim=1), round(weight.size(1) * self.tau))[0][-1]

            prune.custom_from_mask(self.mlp[i], name="weight", mask=ThresholdPruning(threshold).compute_mask(weight, None))

            h = F.relu(self.mlp[i](h))
            layer_masks.append((torch.abs(weight) >= threshold).float())

        prune.custom_from_mask(self.mlp[-1], name="weight", mask=torch.ones_like(self.mlp[-1].weight))
        weight = self.mlp[-1].weight
        threshold = torch.topk(torch.abs(weight).mean(dim=1), round(weight.size(1) * self.tau))[0][-1]
        prune.custom_from_mask(self.mlp[-1], name="weight", mask=ThresholdPruning(threshold).compute_mask(weight, None))

        h = F.softmax(self.mlp[-1](h), dim=1)
        layer_masks.append((torch.abs(weight) >= threshold).float())

        return h, layer_masks
