import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트
from tqdm import tqdm
from tqdm import trange
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
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_output_channels(self):  # todo add output channels when last conv
        channels = []
        last_conv_out_channels = None
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                channels.append(layer.in_channels)
                last_conv_out_channels = layer.out_channels
        # if last_conv_out_channels is not None:
        #     channels.append(last_conv_out_channels)
        return channels

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
        self.policy_net.append(nn.Linear(3 * 32 * 32, 64))
        self.policy_net.append(nn.Linear(64 * 32 * 32, 64))
        self.policy_net.append(nn.Linear(64 * 32 * 32, 128))

        self.policy_net.to(self.device)

        self.channels = self.cnn.get_output_channels()

    def forward(self, x):
        channels_cumsum = np.cumsum(self.channels)
        policies = []
        sample_probs = []
        layer_masks = []

        x = x.to(self.device)
        # u = torch.ones(x.shape[0], 64, x.shape[2], x.shape[3]).to(self.device)
        h = x
        u = torch.ones(h.shape[0], h.shape[1], h.shape[2], h.shape[3]).to(self.device)
        # 첫 번째 Conv 레이어와 마스킹
        # print(f"After first conv: {h.shape}")

        h_re = F.interpolate(h, size=(32, 32))
        p_i = self.policy_net[0](h_re.view(h_re.size(0), -1))
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
        #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
        u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)
        # u_i = u_i[:, channels_cumsum[0]:channels_cumsum[1], :, :]  # 채널을 맞추기 위해 차원을 추가
        # print(f"u_i after first conv: {u_i.shape}")
        # print(f"u after first conv: {u.shape}")
        # h = (h * u) * u_i
        h_next = F.relu(self.cnn.conv1(h * u)) * u_i
        h = h_next
        u = u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        h_re = F.interpolate(h, size=(32, 32))
        p_i = self.policy_net[1](h_re.view(h_re.size(0), -1))
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
        #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
        u_is = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16).to(self.device)
        u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)

        # 두 번째 Conv 레이어와 마스킹
        h_next = F.relu(self.cnn.conv2(h * u)) * u_i

        h = h_next
        u = u_is

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        h = self.cnn.pool(h)

        h_re = F.interpolate(h, size=(32, 32))
        p_i = self.policy_net[2](h_re.view(h_re.size(0), -1))
        p_i = F.sigmoid(p_i)
        p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)
        # u_i = F.interpolate(u_i.unsqueeze(1), size=h.size(1) * h.size(2) * h.size(3), mode='linear',
        #                     align_corners=True).squeeze(1).view(h.size(0), h.size(1), h.size(2), h.size(3))
        u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2), h.size(3)).to(self.device)
        # u = F.interpolate(u, size=(16, 16))

        # 두 번째 Conv 레이어와 마스킹
        h_next = F.relu(self.cnn.conv3(h * u)) * u_i

        h = h_next
        u = u_i

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        h = F.relu(self.cnn.conv4(h))
        h = self.cnn.pool(h)

        # FC1 레이어와 마스킹
        h = h.view(h.size(0), -1)
        h = F.relu(self.cnn.fc1(h))
        # print(f"After FC1: {h.shape}")

        # FC2 레이어와 마스킹
        h = self.cnn.dropout(h)
        h = F.relu(self.cnn.fc2(h))
        # print(f"After FC2: {h.shape}")

        # FC3 레이어와 마스킹
        h = self.cnn.dropout(h)
        h = self.cnn.fc3(h)
        # print(f"After FC3: {h.shape}")

        h = F.softmax(h, dim=1)

        return h, policies, sample_probs, layer_masks

def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=4.27)
    args.add_argument('--lambda_v', type=float, default=0.001)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=50)
    args.add_argument('--condnet_min_prob', type=float, default=1e-3)
    args.add_argument('--condnet_max_prob', type=float, default=1 - 1e-3)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=60)
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

    wandb.init(project="cond_cnn_cifar10",
                config=args.__dict__,
                name='cond_cnn_cifar10_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau)
                )

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
    mlp_optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    # mlp_optimizer = optim.SGD(model.parameters(), lr=learning_rate,
    #                       momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.Adam(model.policy_net.parameters(), lr=0.0003, weight_decay=1e-4)

    # run for 50 epochs
    for epoch in trange(max_epochs):

        model.train()
        costs = 0
        accs = 0
        PGs = 0
        taus = 0
        Ls = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(tqdm(train_loader, 0)):
            mlp_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고

            # 순전파 + 역전파 + 최적화를 한 후
            outputs, policies, sample_probs, layer_masks = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels.to(model.device))
            # Compute the regularization loss L

            policy_flat = torch.cat(policies, dim=1)
            Lb_ = torch.norm(policy_flat.mean(axis=0) - torch.tensor(tau).to(model.device), p=2)
            Le_ = torch.norm(policy_flat.mean(axis=1) - torch.tensor(tau).to(model.device), p=2) / len(policies)

            # Lv_ = -torch.pow(policy_flat - policy_flat.mean(axis=0),2).mean(axis=0).sum()
            Lv_ = -torch.norm(policy_flat - policy_flat.mean(axis=0), p=2, dim=0).sum()

            L = c + lambda_s * (Lb_ + Le_)
            # (torch.pow(torch.cat(policies, dim=1).mean(axis=0) - torch.tensor(tau).to(model.device), 2).mean() +
            #                 torch.pow(torch.cat(policies, dim=1).mean(axis=2) - t

            L += lambda_v * (Lv_)
            # (torch.cat(policies,dim=1).to('cpu').var(axis=1).mean() +
            #                    torch.cat(policies,dim=1).to('cpu').var(axis=2).mean())

            # ifzero = []
            # for l in range(len(layer_masks)):
            #     ifzero.append(np.any(layer_masks[l].cpu().detach().numpy().sum(axis=1)==0))
            # if np.any(ifzero):
            #     print(ifzero)
            #     print('waitwaitwait!!')

            # Compute the policy gradient (PG) loss
            logp = torch.log(policy_flat).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            gradient = (c * (-logp)).item()

            PG.backward()  # it needs to be checked [TODO]

            # # gradient에 NaN 및 큰 값 체크
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any():
            #             print(f"{name}의 gradient에 NaN이 존재합니다.")
            #         if torch.max(param.grad) > 1e2:  # 임계값을 조정하세요
            #             print(f"{name}의 gradient가 너무 큽니다: {torch.max(param.grad).item()}")

            mlp_optimizer.step()
            policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            accbf = acc

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            PGs += PG.to('cpu').item()

            # us = torch.cat(layer_masks, dim=1)
            # tau_ = us.mean().detach().item()
            us = [mask.mean().detach().item() for mask in layer_masks]
            tau_ = np.mean(us)
            taus += tau_
            Ls += L.to('cpu').item()
            # wandb log training/batch
            wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': PG.item(),
                       'train/batch_tau': tau_, 'train/batch_loss': L.item(), 'train/batch_Lb': Lb_.item(),
                       'train/batch_Le': Le_.item(), 'train/batch_Lv': Lv_.item(), 'train/batch_gradient': gradient})

            # print PG.item(), and acc with name
            print(
                'Epoch: {}, Batch: {}, Cost: {:.4f}, PG:{:.5f}, Acc: {:.3f}, Acc: {:.3f}, Tau: {:.3f}, Lb: {:.3f}, Le: {:.3f}, Lv: {:.8f}, gradient: {:.3f}'.format(
                    epoch, i, c.item(), PG.item(), acc, accbf, tau_, Lb_, Le_, Lv_, gradient))

            # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': taus / bn,
                   'train/epoch_PG': PGs / bn, 'train/epoch_L': Ls / bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0
        PGs = 0

        model.eval()
        with torch.no_grad():
            # calculate accuracy on test set
            accs = 0
            bn = 0
            taus = 0
            costs = 0
            PGs = 0
            Ls = 0
            gradients = 0
            Lb_s = 0
            Le_s = 0
            Lv_s = 0
            for i, data in enumerate(test_loader, 0):
                bn += 1
                # get batch
                inputs, labels = data

                # make one hot vector
                y_batch_one_hot = torch.zeros(labels.shape[0], 10)
                y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1, ).tolist()] = 1

                # get output
                outputs, policies, sample_probs, layer_masks = model(torch.tensor(inputs))

                # calculate accuracy
                pred = torch.argmax(outputs, dim=1).to('cpu')
                acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                c = C(outputs, labels.to(model.device))

                # Compute the regularization loss L
                policy_flat = torch.cat(policies, dim=1)
                Lb_ = torch.norm(policy_flat.mean(axis=0) - torch.tensor(tau).to(model.device), p=2)
                Le_ = torch.norm(policy_flat.mean(axis=1) - torch.tensor(tau).to(model.device), p=2) / len(policies)

                # Lv_ = -torch.pow(policy_flat - policy_flat.mean(axis=0),2).mean(axis=0).sum()
                Lv_ = -torch.norm(policy_flat - policy_flat.mean(axis=0), p=2, dim=0).sum()

                L = c + lambda_s * (Lb_ + Le_)
                # (torch.pow(torch.cat(policies, dim=1).mean(axis=0) - torch.tensor(tau).to(model.device), 2).mean() +
                #                 torch.pow(torch.cat(policies, dim=1).mean(axis=2) - t

                L += lambda_v * (Lv_)
                # (torch.cat(policies,dim=1).to('cpu').var(axis=1).mean() +
                #                    torch.cat(policies,dim=1).to('cpu').var(axis=2).mean())

                # ifzero = []
                # for l in range(len(layer_masks)):
                #     ifzero.append(np.any(layer_masks[l].cpu().detach().numpy().sum(axis=1)==0))
                # if np.any(ifzero):
                #     print(ifzero)
                #     print('waitwaitwait!!')

                # Compute the policy gradient (PG) loss
                logp = torch.log(policy_flat).sum(axis=1).mean()
                PG = lambda_pg * c * (-logp) + L

                gradient = (c * (-logp)).item()

                # wandb log test/batch
                # wandb.log({'test/batch_acc': acc, 'test/batch_cost': c.to('cpu').item(), 'test/batch_pg': PG.to('cpu').item()})

                # addup loss and acc
                costs += c.to('cpu').item()
                accs += acc
                PGs += PG.to('cpu').item()
                Ls += L.to('cpu').item()
                Lb_s += Lb_.to('cpu').item()
                Le_s += Le_.to('cpu').item()
                Lv_s += Lv_.to('cpu').item()
                gradients += gradient

                # us = torch.cat(layer_masks, dim=1)
                # tau_ = us.mean().detach().item()
                us = [mask.mean().detach().item() for mask in layer_masks]
                tau_ = np.mean(us)
                taus += tau_
            # print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log test/epoch
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn, 'test/epoch_PG': PGs / bn,
                       'test/epoch_tau': taus / bn, 'test/epoch_L': Ls / bn, 'test/epoch_Lb': Lb_s / bn,
                       'test/epoch_Le': Le_s / bn, 'test/epoch_Lv': Lv_s / bn, 'test/epoch_gradient': gradients / bn})
        torch.save(model.state_dict(),
                   './cond_cnn_cifar10_' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(
                       args.tau) + dt_string + '.pt')
    wandb.finish()


if __name__ == '__main__':
    main()