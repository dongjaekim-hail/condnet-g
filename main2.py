import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Dataset
import lightning as L
from torchmetrics.functional import accuracy
from datasets import load_from_disk
import torchvision.models as models

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
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

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

class LitClassifier(L.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.classifier(x)
        loss = F.cross_entropy(y_hat[0], y)
        acc = accuracy(torch.argmax(y_hat[0],dim = 1), y, task = 'multiclass', num_classes=1000)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

BATCH_SIZE = 50

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dir2save = 'D:/imagenet-1k/'
# dir2save = '/Users/dongjaekim/Documents/imagenet'

data_module = ImageNetDataModule(data_dir=dir2save, batch_size=BATCH_SIZE, debug=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet = ResNet50(device)
resnet = resnet.to(device)
for param in resnet.parameters():
    param.requires_grad = True

model = LitClassifier(resnet)

# train model
trainer = L.Trainer()
trainer.fit(model=model, datamodule=data_module)

print('')


