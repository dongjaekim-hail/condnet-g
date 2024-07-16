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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn.init as init


class model_condnet(nn.Module):
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
        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob

        self.mlp_nlayer = 0

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(self.input_dim, mlp_hidden[0]))
        for i in range(nlayers):
            self.mlp.append(nn.Linear(mlp_hidden[i], mlp_hidden[i+1]))
        self.mlp.append(nn.Linear(mlp_hidden[nlayers], output_dim))
        self.mlp.to(self.device)

        # DOWNSAMPLE
        self.avg_poolings = nn.ModuleList()
        pool_hiddens = [512, *mlp_hidden]
        for i in range(len(self.mlp)):
            stride = round(pool_hiddens[i] / pool_hiddens[i + 1])
            self.avg_poolings.append(nn.AvgPool1d(kernel_size=stride, stride=stride))

        # UPSAMPLE
        self.upsample = nn.ModuleList()
        for i in range(len(self.mlp)):
            stride = round(pool_hiddens[i + 1] / 1024)
            self.upsample.append(nn.Upsample(scale_factor=stride, mode='nearest'))

        # HANDCRAFTED POLICY NET
        # self.policy_net = nn.ModuleList()
        # self.policy_net.append(nn.Linear(28 * 28, 512))
        # self.policy_net.append(nn.Linear(512, 256))
        # self.policy_net.append(nn.Linear(256, 10))

        # self.policy_net.to(self.device)

        # # HANDCRAFTED POLICY NET
        n_each_policylayer = 1
        # n_each_policylayer = 1 # if you have only 1 layer perceptron for policy net
        self.policy_net = nn.ModuleList()
        temp = nn.ModuleList()
        # temp.append(nn.Linear(self.input_dim, mlp_hidden[0])) # BEFORE LARGE MODEL'S
        temp.append(nn.Linear(self.input_dim,  mlp_hidden[0]))
        self.policy_net.append(temp)

        for i in range(len(self.mlp) - 1):
            temp = nn.ModuleList()
            for j in range(n_each_policylayer):
                temp.append(nn.Linear(self.mlp[i].out_features, self.mlp[i+1].out_features)) # BEFORE LARGE MODEL'S
                # temp.append(nn.Linear(self.mlp[i].out_features, mlp_hidden[i]))
            self.policy_net.append(temp)
        self.policy_net.to(self.device)

    def forward(self, x):
        # return policies
        policies = []
        sample_probs = []
        layer_masks = []
        x = x.view(-1, self.input_dim).to(self.device)

        # param_min = 0
        # param_max = 0
        # # Initial check for NaNs in model parameters
        # for name, param in self.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"NaN detected in {name} ")
        #     if param_max < param.max():
        #         param_max = param.max().item()
        #     if param_min > param.min():
        #         param_min = param.min().item()
        # # print('param_min:', param_min, 'param_max', param_max)

        # for each layer
        h = x
        u = torch.ones(h.shape[0], h.shape[1]).to(self.device)

        for i in range(len(self.mlp)-1):

            # param_min = 0
            # param_max = 0
            # # Initial check for NaNs in model parameters
            # for name, param in self.policy_net[i][0].named_parameters():
            #     if torch.isnan(param).any():
            #         print(f"NaN detected in {name}")
            #     if param_max < param.max():
            #         param_max = param.max().item()
            #     if param_min > param.min():
            #         param_min = param.min().item()
            # # print('param_min:', param_min, 'param_max', param_max)


            h_clone = h.clone()
            p_is = self.policy_net[i][0](h_clone.detach())
            # p_i = self.policy_net[i][0](h)
            # Check for NaNs after first policy net layer
            if torch.isnan(p_is).any():
                print(f"NaN detected in policy_net[{i}][0] output")

            p_i = F.sigmoid(p_is)

            # Check for NaNs after sigmoid activation
            if torch.isnan(p_i).any():
                print(f"NaN detected after sigmoid(policy_net[{i}][0] output)")

            for j in range(1, len(self.policy_net[i])):
                p_is = self.policy_net[i][j](p_i)
                p_i = F.sigmoid(p_is)

            # p_i = p_i * (self.condnet_max_prob - self.condnet_min_prob) + self.condnet_min_prob
            p_i = torch.clamp(p_i, min=self.condnet_min_prob, max=self.condnet_max_prob)

            if np.any(np.isnan(p_i.cpu().detach().numpy())):
                print('wait a sec')

            # # print(p_i.max().item(), p_i.min().item())
            # invalid_values = p_i[(p_i < 0) | (p_i > 1)]
            # if invalid_values.numel() > 0:
            #     print("Invalid values in p_i:", invalid_values)

            u_i = torch.bernoulli(p_i).to(self.device)

            # debug[TODO]
            # u_i = torch.ones(u_i.shape[0], u_i.shape[1])

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size = (1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1-p_i) * (1-u_i)

            # idx = torch.where(u_i == 0)[0]

            # h_next = F.relu(self.mlp[i](h*u.detach()))*u_i
            h_next = F.relu(self.mlp[i](h*u))*u_i
            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

        # last layer just go without dynamic sampling
        h = self.mlp[-1](h)
        h = F.softmax(h, dim=1)


        return h, policies, sample_probs, layer_masks


def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=1)
    args.add_argument('--lambda_s', type=float, default=10)
    args.add_argument('--lambda_v', type=float, default=0.5)
    args.add_argument('--lambda_l2', type=float, default=1e-5)
    args.add_argument('--lambda_pg', type=float, default=1e-4)
    args.add_argument('--tau', type=float, default=0.3)
    args.add_argument('--max_epochs', type=int, default=50)
    args.add_argument('--condnet_min_prob', type=float, default=0.2)
    args.add_argument('--condnet_max_prob', type=float, default=0.8)
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
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 평균을 0.5로, 표준편차를 0.5로 하여 [-1, 1] 범위로 변환
        ])
    )


    test_dataset = datasets.MNIST(
        root="../data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 평균을 0.5로, 표준편차를 0.5로 하여 [-1, 1] 범위로 변환
        ])
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
                name='cond_lastchance1024' + '_tau=' + str(args.tau)
                )

    # create model
    model = model_condnet(args)
    # model = model_condnet2()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in model.mlp.parameters():
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

            policy_flat = torch.cat(policies,dim=1)
            Lb_ = torch.pow(policy_flat.mean(axis=0)-torch.tensor(tau).to(model.device),2).sqrt().sum()
            Le_ = torch.pow(policy_flat.mean(axis=1)-torch.tensor(tau).to(model.device),2).sqrt().mean()

            Lv_ = -torch.pow(policy_flat - policy_flat.mean(axis=0),2).mean(axis=0).sum()

            L = c + lambda_s * (Lb_ +Le_)
                # (torch.pow(torch.cat(policies, dim=1).mean(axis=0) - torch.tensor(tau).to(model.device), 2).mean() +
                #                 torch.pow(torch.cat(policies, dim=1).mean(axis=2) - t

            L += lambda_v * (-1) * (Lv_)
                 # (torch.cat(policies,dim=1).to('cpu').var(axis=1).mean() +
                 #                    torch.cat(policies,dim=1).to('cpu').var(axis=2).mean())

            # Compute the policy gradient (PG) loss
            logp = torch.log(policy_flat).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            PG.backward() # it needs to be checked [TODO]

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

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            PGs += PG.to('cpu').item()

            # wandb log training/batch
            wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_pg': PG.item(), 'train/batch_tau': tau})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.10f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), acc, np.mean([tau_.mean().item() for tau_ in layer_masks])
                                                                                                     ))

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
                policy_flat = torch.cat(policies,dim=1)
                Lb_ = torch.pow(policy_flat.mean(axis=0)-torch.tensor(tau).to(model.device),2).sqrt().sum()
                Le_ = torch.pow(policy_flat.mean(axis=1)-torch.tensor(tau).to(model.device),2).sqrt().mean()

                Lv_ = -torch.pow(policy_flat - policy_flat.mean(axis=0),2).mean(axis=0).sum()

                L = c + lambda_s * (Lb_ +Le_)
                    # (torch.pow(torch.cat(policies, dim=1).mean(axis=0) - torch.tensor(tau).to(model.device), 2).mean() +
                    #                 torch.pow(torch.cat(policies, dim=1).mean(axis=2) - t

                L += lambda_v  * (Lv_)
                     # (torch.cat(policies,dim=1).to('cpu').var(axis=1).mean() +
                     #                    torch.cat(policies,dim=1).to('cpu').var(axis=2).mean())

                # Compute the policy gradient (PG) loss
                logp = torch.log(policy_flat).sum(axis=1).mean()
                PG = lambda_pg * c * (-logp) + L

                # wandb log test/batch
                # wandb.log({'test/batch_acc': acc, 'test/batch_cost': c.to('cpu').item(), 'test/batch_pg': PG.to('cpu').item()})

                # addup loss and acc
                costs += c.to('cpu').item()
                accs += acc
                PGs += PG.to('cpu').item()

                tau_ = (policy_flat).mean().detach().item()
                taus += tau_
            #print accuracy
            print('Test Accuracy: {}'.format(accs / bn))
            # wandb log test/epoch
            wandb.log({'test/epoch_acc': accs / bn, 'test/epoch_cost': costs / bn, 'test/epoch_pg': PGs / bn, 'test/epoch_tau': taus / bn })
        torch.save(model.state_dict(), './cond1024_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
    wandb.finish()
if __name__=='__main__':
    main()