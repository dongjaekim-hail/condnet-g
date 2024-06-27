import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

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

class model_condnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.cnn = SimpleCNN().to(self.device)

        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob

        # HANDCRAFTED POLICY NET
        self.policy_net = nn.ModuleList()
        self.policy_net.append(nn.Linear(64 * 32 * 32, 1024))  # Updated for new conv layers output size
        self.policy_net.append(nn.Linear(64 * 16 * 16, 1024))   # After pooling
        self.policy_net.append(nn.Linear(256, 1024))          # FC1 레이어
        self.policy_net.append(nn.Linear(256, 1024))          # FC2 레이어
        self.policy_net.append(nn.Linear(10, 1024))           # FC3 레이어

        self.policy_net.to(self.device)

    def forward(self, x):
        policies = []
        sample_probs = []
        layer_masks = []

        x = x.to(self.device)
        u = torch.ones(x.shape[0], 64, x.shape[2], x.shape[3]).to(self.device)

        # 첫 번째 Conv 레이어와 마스킹
        h = F.relu(self.cnn.conv1(x))
        # print(f"After first conv: {h.shape}")

        p_i = self.policy_net[0](h.view(h.size(0), -1))
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
                            align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
        # print(f"u_i after first conv: {u_i.shape}")
        # print(f"u after first conv: {u.shape}")
        h = (h * u) * u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        u = u_i

        # 두 번째 Conv 레이어와 마스킹
        h = F.relu(self.cnn.conv2(h))
        h = self.cnn.pool(h)
        # print(f"After second conv and pooling: {h.shape}")

        p_i = self.policy_net[1](h.view(h.size(0), -1))
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
                            align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
        # print(f"u_i after second conv: {u_i.shape}")
        # print(f"u after second conv: {u.shape}")
        u = F.interpolate(u.unsqueeze(1), size=(h.size(1), h.size(2), h.size(3)), mode='nearest').squeeze(1)
        h = (h * u) * u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        u = u_i

        # FC1 레이어와 마스킹
        h = h.view(h.size(0), -1)
        h = F.relu(self.cnn.fc1(h))
        # print(f"After FC1: {h.shape}")

        p_i = self.policy_net[2](h)
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        # print(f"u_i after fc1: {u_i.shape}")
        # print(f"u after fc1: {u.shape}")
        u_i = F.interpolate(u_i.unsqueeze(1), size=256, mode='linear', align_corners=True).squeeze(1)
        u = F.interpolate(u.view(u.size(0), -1).unsqueeze(1), size=h.size(1), mode='linear').squeeze(1)
        h = (h * u) * u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        u = u_i

        # FC2 레이어와 마스킹
        h = F.relu(self.cnn.fc2(h))
        # print(f"After FC2: {h.shape}")

        p_i = self.policy_net[3](h)
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        # print(f"u_i after fc2: {u_i.shape}")
        # print(f"u after fc2: {u.shape}")
        u_i = F.interpolate(u_i.unsqueeze(1), size=256, mode='linear', align_corners=True).squeeze(1)
        u = F.interpolate(u.view(u.size(0), -1).unsqueeze(1), size=h.size(1), mode='linear').squeeze(1)
        h = (h * u) * u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        u = u_i

        # FC3 레이어와 마스킹
        h = self.cnn.dropout(h)
        h = self.cnn.fc3(h)
        # print(f"After FC3: {h.shape}")

        p_i = self.policy_net[4](h)
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        # print(f"u_i after fc3: {u_i.shape}")
        # print(f"u after fc3: {u.shape}")
        u_i = F.interpolate(u_i.unsqueeze(1), size=10, mode='linear', align_corners=True).squeeze(1)
        u = F.interpolate(u.view(u.size(0), -1).unsqueeze(1), size=h.size(1), mode='linear').squeeze(1)
        h = (h * u) * u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        h = F.softmax(h, dim=1)

        return h, policies, sample_probs, layer_masks

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
    args.add_argument('--max_epochs', type=int, default=200)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--lr', type=float, default=0.01)
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
    train_dataset = datasets.CIFAR10(
        root="../data/mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.CIFAR10(
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

    # wandb.init(project="condgnet",
    #             config=args.__dict__,
    #             name='cond_lastchance1024' + '_tau=' + str(args.tau)
    #             )

    # create model
    model = model_condnet(args)
    # model = model_condnet2()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.cnn.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.policy_net.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)
    # mlp_optimizer = optim.SGD(model.mlp.parameters(), lr=learning_rate,
    #                       momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(model.policy_net.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=lambda_l2)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0
        PGs = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(train_loader, 0):
            mlp_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고

            # 순전파 + 역전파 + 최적화를 한 후
            outputs, policies, sample_probs, layer_masks  = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels.to(model.device))
            # Compute the regularization loss L

            L = c + lambda_s * (torch.pow(torch.stack(policies).mean(axis=1) - torch.tensor(tau).to(model.device), 2).mean() +
                                torch.pow(torch.stack(policies).mean(axis=2) - torch.tensor(tau).to(model.device), 2).mean())

            L += lambda_v * (-1) * (torch.stack(policies).to('cpu').var(axis=1).mean() +
                                    torch.stack(policies).to('cpu').var(axis=2).mean())



            # Compute the policy gradient (PG) loss
            logp = torch.log(torch.cat(policies)).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            PG.backward() # it needs to be checked [TODO]
            mlp_optimizer.step()
            policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            PGs += PG.to('cpu').item()

            # wandb log training/batch
            # wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': PG.item(), 'train/batch_tau': tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, np.mean([tau_.mean().item() for tau_ in layer_masks])
                                                                                                     ))

        # wandb log training/epoch
        # wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': tau, 'train/epoch_PG': PGs/bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0
        PGs = 0

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
                outputs, policies, sample_probs, layer_masks = model(torch.tensor(inputs))

                # calculate accuracy
                pred = torch.argmax(outputs, dim=1).to('cpu')
                acc  = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                c = C(outputs, labels.to(model.device))

                # Compute the regularization loss L

                L = c + lambda_s * (torch.pow(torch.stack(policies).mean(axis=1) - torch.tensor(tau).to(model.device), 2).mean() +
                                    torch.pow(torch.stack(policies).mean(axis=2) - torch.tensor(tau).to(model.device), 2).mean())

                L += lambda_v * (-1) * (torch.stack(policies).var(axis=1).mean() +
                                        torch.stack(policies).var(axis=2).mean())



                # Compute the policy gradient (PG) loss
                logp = torch.log(torch.cat(policies)).sum(axis=1).mean()
                PG = lambda_pg * c * (-logp) + L

                # wandb log test/batch
                # wandb.log({'test/batch_acc': acc, 'test/batch_cost': c.to('cpu').item(), 'test/batch_pg': PG.to('cpu').item()})

                # addup loss and acc
                costs += c.to('cpu').item()
                accs += acc
                PGs += PG.to('cpu').item()

                tau_ = torch.stack(policies).mean().detach().item()
                taus += tau_
            #print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log test/epoch
            # wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn, 'test/epoch_pg': PGs / bn, 'test/epoch_tau': taus / bn })
        torch.save(model.state_dict(), './cond1024_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
    # wandb.finish()
if __name__=='__main__':
    main()