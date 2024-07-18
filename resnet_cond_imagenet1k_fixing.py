import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import random

from datetime import datetime

def he_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def random_resize(image):
    # 이미지의 짧은 쪽 길이를 [256, 480] 범위에서 무작위로 샘플링하여 크기를 조정합니다.
    size = random.randint(256, 480)
    # 이미지를 유지하면서 짧은 쪽이 size가 되도록 크기를 조정합니다.
    return transforms.Resize(size)(image)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet_model = models.resnet18(pretrained=True)
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

    def get_output_channels(self):
        channels = []
        for i in range(len(self.modules[0:4])):
            try:
                channels.append(self.modules[0:4][i].in_channels)
            except Exception:
                continue
        for layer in self.modules[4:8]:
            for bottleneck in layer:
                try:
                    channels.append(bottleneck.conv1.in_channels)
                    channels.append(bottleneck.conv2.in_channels)
                    # if bottleneck.downsample is not None:
                    #     for downsample_layer in bottleneck.downsample:
                    #         if isinstance(downsample_layer, nn.Conv2d):
                    #             num_channels_ls.append(downsample_layer.in_channels)
                except Exception:
                    continue
        return channels

class model_condnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.resnet = ResNet50().to(self.device)

        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob

        # self.to_be_cond = []
        #
        # self.resnet.conv1
        # for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
        #     if 'conv' in str(layer).lower():
        #         self.to_be_cond.append(layer)

        # HANDCRAFTED POLICY NET
        self.policy_net = nn.ModuleList()
        self.policy_net.append(nn.Linear(3 * 7 * 7, 64 * 7 * 7))

        self.policy_net.append(nn.Linear(64 * 7 * 7, 64 * 7 * 7))
        self.policy_net.append(nn.Linear(64 * 7 * 7, 64 * 7 * 7))
        self.policy_net.append(nn.Linear(64 * 7 * 7, 64 * 7 * 7))
        self.policy_net.append(nn.Linear(64 * 7 * 7, 64 * 7 * 7))

        self.policy_net.append(nn.Linear(64 * 7 * 7, 128 * 7 * 7))
        self.policy_net.append(nn.Linear(128 * 7 * 7, 128 * 7 * 7))
        self.policy_net.append(nn.Linear(128 * 7 * 7, 128 * 7 * 7))
        self.policy_net.append(nn.Linear(128 * 7 * 7, 128 * 7 * 7))

        self.policy_net.append(nn.Linear(128 * 7 * 7, 256 * 7 * 7))
        self.policy_net.append(nn.Linear(256 * 7 * 7, 256 * 7 * 7))
        self.policy_net.append(nn.Linear(256 * 7 * 7, 256 * 7 * 7))
        self.policy_net.append(nn.Linear(256 * 7 * 7, 256 * 7 * 7))

        self.policy_net.to(self.device)

        self.channels = self.resnet.get_output_channels()

        self.resnet.apply(he_init)

    def forward(self, x):
        channels_cumsum = np.cumsum(self.channels)
        policies = []
        sample_probs = []
        layer_masks = []
        i = 0

        x = x.to(self.device)
        h = x
        u = torch.ones(h.shape[0], h.shape[1], h.shape[2], h.shape[3]).to(self.device)

        # ResNet50의 첫 번째 레이어
        h_clone = h.clone()
        h_re = F.interpolate(h_clone, size=(7, 7))
        p_is = self.policy_net[0](h_re.view(h.size(0), -1))
        if torch.isnan(p_is).any():
            print(f"NaN detected in policy_net[{i}][0] output")

        p_i = torch.sigmoid(p_is)
        if torch.isnan(p_i).any():
            print(f"NaN detected after sigmoid(policy_net[{i}][0] output)")

        p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)
        if np.any(np.isnan(p_i.cpu().detach().numpy())):
            print('wait a sec')

        u_i = torch.bernoulli(p_i).to(self.device)

        if u_i.sum() == 0:
            idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
            u_i[idx] = 1

        sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)

        u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2)//2, h.size(3)//2).to(self.device)
        u_i = u_i[:, channels_cumsum[0]:channels_cumsum[1], :, :]  # 채널을 맞추기 위해 차원을 추가
        h_next = F.relu(self.resnet.bn1(self.resnet.conv1(h * u))) * u_i
        h = h_next
        u = u_i

        # if u_i is in shape of (B, feature numbers)
        # u_i_expaneded = u_i.unsqueeze(-1).unsqueeze(-1).expand(B,C,H,W).to(device)
        # h = h*u_i_expaneded

        policies.append(p_i)
        sample_probs.append(sampling_prob)
        layer_masks.append(u_i)

        h = self.resnet.max_pool(h)
        # print(f"maxpool output size: {h.size()}")

        policy_index = 1
        # 각 레이어의 bottleneck 블록을 통과한 특징
        # for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
        for layer_idx, layer in enumerate([self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]):
            for bottleneck in layer:
                residual = h

                if layer_idx != 3:
                    h_clone = h.clone()
                    h_re = F.interpolate(h_clone, size=(7, 7))
                    p_is = self.policy_net[i + 1](h_re.view(h_re.size(0), -1))
                    if torch.isnan(p_is).any():
                        print(f"NaN detected in policy_net[{i + 1}][0] output")

                    p_i = torch.sigmoid(p_is)
                    if torch.isnan(p_i).any():
                        print(f"NaN detected after sigmoid(policy_net[{i + 1}][0] output)")

                    p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)
                    if np.any(np.isnan(p_i.cpu().detach().numpy())):
                        print('wait a sec')

                    u_i = torch.bernoulli(p_i).to(self.device)

                    if u_i.sum() == 0:
                        idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                        u_i[idx] = 1

                    sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)

                    u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2)//2, h.size(3)//2).to(self.device)
                    u_i = u_i[:, channels_cumsum[i + 1]:channels_cumsum[i + 2], :, :]  # 채널을 맞추기 위해 차원을 추가
                    out_next = F.relu(bottleneck.bn1(bottleneck.conv1(h * u))) * u_i
                    out = out_next
                    u = u_i

                    # if u_i is in shape of (B, feature numbers)
                    # u_i_expaneded = u_i.unsqueeze(-1).unsqueeze(-1).expand(B,C,H,W).to(device)
                    # h = h*u_i_expaneded
                    i += 1

                    policies.append(p_i)
                    sample_probs.append(sampling_prob)
                    layer_masks.append(u_i)

                    out_clone = out.clone()
                    out_re = F.interpolate(out_clone, size=(7, 7))
                    p_is = self.policy_net[i + 1](out_re.view(out_re.size(0), -1))
                    if torch.isnan(p_is).any():
                        print(f"NaN detected in policy_net[{i + 1}][0] output")

                    p_i = torch.sigmoid(p_is)
                    if torch.isnan(p_i).any():
                        print(f"NaN detected after sigmoid(policy_net[{i + 1}][0] output)")

                    p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)
                    if np.any(np.isnan(p_i.cpu().detach().numpy())):
                        print('wait a sec')

                    u_i = torch.bernoulli(p_i).to(self.device)

                    if u_i.sum() == 0:
                        idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                        u_i[idx] = 1

                    sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)

                    u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2)//2, h.size(3)//2).to(self.device)
                    u_i = u_i[:, channels_cumsum[i + 1]:channels_cumsum[i + 2], :, :]  # 채널을 맞추기 위해 차원을 추가

                    out = bottleneck.bn2(bottleneck.conv2(out))
                    # print(f"layer bottleneck.conv3 output size: {out.size()}")
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(h)
                    out += residual
                    out_next = bottleneck.relu(out * u) * u_i
                    out = out_next
                    u = u_i

                    # if u_i is in shape of (B, feature numbers)
                    # u_i_expaneded = u_i.unsqueeze(-1).unsqueeze(-1).expand(B,C,H,W).to(device)
                    # h = h*u_i_expaneded
                    i += 1

                    policies.append(p_i)
                    sample_probs.append(sampling_prob)
                    layer_masks.append(u_i)

                    h = out

                else:
                    # layer4의 conv1 처리
                    h_clone = h.clone()
                    h_re = F.interpolate(h_clone, size=(7, 7))
                    p_is = self.policy_net[i + 1](h_re.view(h_re.size(0), -1))
                    if torch.isnan(p_is).any():
                        print(f"NaN detected in policy_net[{i + 1}][0] output")

                    p_i = torch.sigmoid(p_is)
                    if torch.isnan(p_i).any():
                        print(f"NaN detected after sigmoid(policy_net[{i + 1}][0] output)")

                    p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)
                    if np.any(np.isnan(p_i.cpu().detach().numpy())):
                        print('wait a sec')

                    u_i = torch.bernoulli(p_i).to(self.device)

                    if u_i.sum() == 0:
                        idx = np.random.uniform(0, u_i.shape[0], size=(1)).astype(np.int16)
                        u_i[idx] = 1

                    sampling_prob = p_i * u_i + (1 - p_i) * (1 - u_i)

                    u_i = u_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.size(2) // 2, h.size(3) // 2).to(self.device)
                    u_i = u_i[:, channels_cumsum[i + 1]:channels_cumsum[i + 2], :, :]  # 채널을 맞추기 위해 차원을 추가

                    out_next = F.relu(bottleneck.bn1(bottleneck.conv1(h * u))) * u_i
                    out = out_next
                    u = u_i

                    policies.append(p_i)
                    sample_probs.append(sampling_prob)
                    layer_masks.append(u_i)

                    i += 1

                    # layer4의 conv2 처리
                    out = bottleneck.bn2(bottleneck.conv2(out))
                    if bottleneck.downsample is not None:
                        residual = bottleneck.downsample(h)
                    out += residual
                    out = bottleneck.relu(out)
                    h = out

        h = self.resnet.avg_pool(h)
        h = torch.flatten(h, 1)
        h = self.resnet.fc(h)

        return h, policies, sample_probs, layer_masks

def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=7)
    args.add_argument('--lambda_v', type=float, default=1.5)
    args.add_argument('--lambda_l2', type=float, default=0.0001)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--tau', type=float, default=0.6)
    args.add_argument('--max_epochs', type=int, default=3000)
    args.add_argument('--condnet_min_prob', type=float, default=0.1)
    args.add_argument('--condnet_max_prob', type=float, default=0.9)
    args.add_argument('--lr', type=float, default=0.1)
    args.add_argument('--BATCH_SIZE', type=int, default=8)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=128)
    args.add_argument('--accum-step', type=int, default=32)
    args = args.parse_args()
    lambda_s = args.lambda_s
    lambda_v = args.lambda_v
    lambda_l2 = args.lambda_l2
    lambda_pg = args.lambda_pg
    tau = args.tau
    learning_rate = args.lr
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE

    dataset_path = r'C:\Users\97dnd\anaconda3\envs\torch\pr\resnet\data'
    dataset = load_from_disk(dataset_path)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: random_resize(image)),  # 무작위 크기 조정
        transforms.RandomCrop(224),  # 224×224 크기의 영역을 무작위로 샘플링합니다.
        transforms.RandomHorizontalFlip(),  # 이미지의 수평 반전을 무작위로 적용합니다.
        transforms.Grayscale(num_output_channels=3),  # 그레이스케일 이미지를 RGB로 변환합니다.
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 픽셀별 평균값을 차감합니다.
        transforms.ColorJitter(),  # 표준 색상 증강 기법을 적용합니다.
    ])

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    class CustomDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image, label = item['image'], item['label']
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = CustomDataset(dataset['validation'], transform=transform)
    test_dataset = CustomDataset(dataset['validation'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    wandb.init(project="condgnetre",
                config=args.__dict__,
                name='resnet_cond_imagenet1k_conv' + '_tau=' + str(args.tau)
                )

    # create model
    model = model_condnet(args)
    # model = model_condnet2()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.resnet.parameters():
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

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(mlp_optimizer, 'min', factor=0.1, patience=10)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0
        PGs = 0
        Ls = 0
        taus = 0

        bn = 0

        mlp_optimizer.zero_grad()
        policy_optimizer.zero_grad()

        L_accum = []
        c_accum = []
        PG_accum = []
        acc_accum = []
        tau_accum = []

        # run for each batch
        for i, data in enumerate(train_loader, 0):

            bn += 1
            # get batch
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고

            # 순전파 + 역전파 + 최적화를 한 후
            outputs, policies, sample_probs, layer_masks  = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 1000)
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

            # PG.backward() # it needs to be checked [TODO]
            # mlp_optimizer.step()
            # policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            L_accum.append(L.item())
            c_accum.append(c.item())
            PG_accum.append(PG.item())
            acc_accum.append(acc)
            tau_accum.append(tau)

            PG /= args.accum_step
            PG.backward()

            if (i + 1) % args.accum_step == 0 or ((i + 1) == len(train_loader)):
                mlp_optimizer.step()
                policy_optimizer.step()

                # addup loss and acc
                costs += np.sum(c_accum)
                accs += np.mean(acc_accum)
                PGs += np.sum(PG_accum)
                Ls += np.sum(L_accum)

                taus += np.mean(tau_accum)

                # print PG.item(), and acc with name
                # print(
                # 'Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Accbf: {:.3f}, Tau: {:.3f}'.format(epoch, i,
                #                                                                                                c.item(),
                #                                                                                                PG.item(),
                #                                                                                                acc,
                #                                                                                                accbf,
                #                                                                                                tau_))
                print(
                    'Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Acc: {:.3f}, Tau: {:.3f}'.format(
                        epoch, i, np.sum(c_accum) / args.accum_step, np.sum(PG_accum) / args.accum_step,
                        np.mean(acc_accum), np.mean(tau_accum)))

                mlp_optimizer.zero_grad()
                policy_optimizer.zero_grad()

                L_accum = []
                c_accum = []
                PG_accum = []
                acc_accum = []
                tau_accum = []

            # # addup loss and acc
            # costs += c.to('cpu').item()
            # accs += acc
            # PGs += PG.to('cpu').item()
            #
            # # wandb log training/batch
            # wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': PG.item(), 'train/batch_tau': tau})
            #
            # # print PG.item(), and acc with name
            # print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, np.mean([tau_.mean().item() for tau_ in layer_masks])
            #                                                                                          ))

        scheduler.step(costs / bn)

        # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_tau': tau, 'train/epoch_PG': PGs/bn})

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
                y_batch_one_hot = torch.zeros(labels.shape[0], 1000)
                y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1,).tolist()] = 1

                # get output
                outputs, policies, sample_probs, layer_masks = model(torch.tensor(inputs))

                # calculate accuracy
                pred = torch.argmax(outputs, dim=1).to('cpu')
                acc  = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 1000)
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
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn, 'test/epoch_pg': PGs / bn, 'test/epoch_tau': taus / bn })
        torch.save(model.state_dict(), './resnet_cond_imagenet1k_conv' + 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
    wandb.finish()
if __name__=='__main__':
    main()