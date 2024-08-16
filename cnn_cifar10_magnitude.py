import torch
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import numpy as np
import wandb

# Setting up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

# CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self, tau):
        super(SimpleCNN, self).__init__()
        self.tau = tau
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1),  # conv1
            nn.Conv2d(64, 64, 3, padding=1),  # conv2
            nn.Conv2d(64, 128, 3, padding=1),  # conv3
            nn.Conv2d(128, 128, 3, padding=1)  # conv4
        ])
        self.pooling_layer = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc_layers = nn.ModuleList([
            nn.Linear(128 * 8 * 8, 256),  # fc1
            nn.Linear(256, 256),  # fc2
            nn.Linear(256, 10)  # fc3
        ])

    def forward(self, x):
        # Forward pass through convolutional layers with magnitude-based pruning
        for i, layer in enumerate(self.conv_layers):
            # calculate pruning threshold
            weight_magnitudes = torch.mean(layer.weight.abs(), dim=(1, 2, 3))
            sorted_weight_magnitudes, _ = torch.sort(weight_magnitudes, descending=True)
            threshold = sorted_weight_magnitudes[round(len(weight_magnitudes) * self.tau)]
            mask = weight_magnitudes >= threshold
            # match the shape of the mask to the weight tensor
            mask = mask.repeat(x.size(0), 1).view(x.size(0), -1, 1, 1)

            if i == len(self.conv_layers): # [TODO]: last layer prune ?
                x = F.relu(layer(x))* mask
            else:
                x = F.relu(layer(x))* mask
            # print(f'Layer {i}''s output shape ', x.shape)
            if i % 2 == 0:
                x = self.pooling_layer(x)
        x = torch.flatten(x, 1)  # Flattening the tensor for fully connected layers

        # Forward pass through fully connected layers
        for i, layer in enumerate(self.fc_layers):
            if i == len(self.fc_layers) - 1:
                x = layer(x) # -> output layer
            else:
                x = F.relu(layer(x))
                x = self.dropout(x)
        return x

# Main training loop
def main():
    # Argument setup
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--max_epochs', type=int, default=10)
    args.add_argument('--lr', type=float, default=0.0003)
    args.add_argument('--BATCH_SIZE', type=int, default=64)
    args.add_argument('--tau', type=float, default=0.6)
    args = args.parse_args()

    # Dataset setup
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    # Model, loss function, optimizer setup
    model = SimpleCNN(args.tau).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    wandb.init(project="condgtest_dk_test", config=args.__dict__, name=f'cnn_magnitude_pruning_{dt_string}')

    # Training loop
    for epoch in range(args.max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).sum().item() / args.BATCH_SIZE

            wandb.log({'train/batch_cost': loss.item(), 'train/batch_acc': acc})
            print(f"Batch loss: {loss.item()}, Batch accuracy: {acc}")
        # Evaluation loop
        model.eval()
        loss_ = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_ += criterion(outputs, labels).item()

        accuracy = correct / total

        wandb.log({'test/epoch_cost': loss_, 'test/epoch_accuracy': accuracy})

    # Save model
    torch.save(model.state_dict(), f'cnn_magnitude_pruned_{dt_string}.pth')
    wandb.finish()

if __name__ == "__main__":
    main()
