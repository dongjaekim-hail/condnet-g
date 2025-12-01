import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

from torch_geometric.nn import DenseSAGEConv

import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login(key="e927f62410230e57c5ef45225bd3553d795ffe01")

class Mlp(nn.Module):
    def __init__(self, gnn_policy = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(28*28, 512))
        self.layers.append(nn.Linear(512, 256))
        self.layers.append(nn.Linear(256, 10))
        self.gnn_policy = gnn_policy

    def forward(self, x, cond_drop=False, us=None):
        hs = [x]
        # flatten
        layer_cumsum = [0]
        for layer in self.layers:
            layer_cumsum.append(layer.in_features)
        layer_cumsum.append(self.layers[-1].out_features)
        layer_cumsum = np.cumsum(layer_cumsum)

        idx = 0
        if not cond_drop:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    # 첫 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    hs.append(x)
                elif i == 1:
                    # 두 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    hs.append(x)
                elif i == 2:
                    # 세 번째 레이어
                    x = layer(x)
                    hs.append(x)

        else:
            if us is None:
                raise ValueError('us should be given')
            # conditional activation
            for i, layer in enumerate(self.layers):
                us = us.squeeze()
                if i == 0:
                    # 첫 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    idx += 1
                elif i == 1:
                    # 두 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]
                    idx += 1
                elif i == 2:
                    # 세 번째 레이어
                    x = layer(x)
                    x = x * us[:, layer_cumsum[idx + 1]:layer_cumsum[idx + 2]]

                hs.append(x)

        # softmax
        x = F.softmax(x, dim=1)
        return x, hs

    def forward_layerwise(self, x, adj_batch=None, cond_drop=False):
        hs = [x]
        # flatten
        layer_cumsum = [0]
        for layer in self.layers:
            layer_cumsum.append(layer.in_features)
        layer_cumsum.append(self.layers[-1].out_features)
        layer_cumsum = np.cumsum(layer_cumsum)
        
        uslist = []
        pslist = []

        
        idx = 0
        if not cond_drop:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    # 첫 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    hs.append(x)
                elif i == 1:
                    # 두 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    hs.append(x)
                elif i == 2:
                    # 세 번째 레이어
                    x = layer(x)
                    hs.append(x)
        else:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    # gnn_policy로부터 얻은 u 사
                    us, ps = self.gnn_policy.forward_layer(hs[-1], batch_adj=adj_batch[i][:,layer_cumsum[idx]:layer_cumsum[idx + 1], layer_cumsum[idx]:layer_cumsum[idx + 1]], li=i)
                    # us = us[:, layer_cumsum[idx]:layer_cumsum[idx + 1]]
                    uslist.append(us)
                    pslist.append(ps)
                    x = x * us.squeeze()
                    # 첫 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    idx += 1
                elif i == 1:
                    us, ps = self.gnn_policy.forward_layer(hs[-1], batch_adj=adj_batch[i][:,layer_cumsum[idx]:layer_cumsum[idx + 1], layer_cumsum[idx]:layer_cumsum[idx + 1]], li=i)
                    uslist.append(us)
                    pslist.append(ps)
                    # us = us[:, layer_cumsum[idx]:layer_cumsum[idx + 1]]
                    x = x * us.squeeze()
                    # 두 번째 레이어
                    x = layer(x)
                    x = F.relu(x)
                    idx += 1
                elif i == 2:
                    us, ps = self.gnn_policy.forward_layer(hs[-1], batch_adj=adj_batch[i][:,layer_cumsum[idx]:layer_cumsum[idx + 1], layer_cumsum[idx]:layer_cumsum[idx + 1]], li=i)
                    uslist.append(us)
                    pslist.append(ps)
                    # us = us[:, layer_cumsum[idx]:layer_cumsum[idx + 1]]
                    x = x * us.squeeze()
                    # 세 번째 레이어
                    x = layer(x)
                hs.append(x)
        # softmax
        x = F.softmax(x, dim=1)
        return x, hs, uslist, pslist
    
    
class GnnBlock(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # 내부적으로는 ModuleList나 개별 변수로 선언
        self.conv1 = DenseSAGEConv(input_dim, hidden_size)
        self.conv2 = DenseSAGEConv(hidden_size, hidden_size)
        self.conv3 = DenseSAGEConv(hidden_size, input_dim) # 다시 원래 차원으로? 혹은 hidden으로? (작성하신 코드 의도 반영)
        self.fc = nn.Linear(input_dim, 1) # 마지막 확률값 생성용

    def forward(self, x, adj):
        # x: [Batch, Nodes, 1] or [Batch, Nodes, Features]
        
        # Conv 레이어들은 adj가 필요함
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = F.relu(self.conv3(x, adj))
        
        # 마지막 Linear는 adj 필요 없음
        x = self.fc(x)
        
        return x
    
class Gnn(nn.Module):
    def __init__(self, hs, minprob, maxprob, hidden_size = 64):
        super().__init__()
        
        self.hs_size = [layer.shape[1] for layer in hs]
        
        self.blocks = nn.ModuleList()
        
        for i in range(len(self.hs_size)):
            # 각 레이어 사이즈에 맞는 Block 생성하여 추가
            # 입력 dim이 1이라고 가정 (코드상 unsqueeze(-1) 하셨으므로)
            self.blocks.append(GnnBlock(input_dim=1, hidden_size=hidden_size)) 
            # *주의: 위 GnnBlock의 입출력 차원은 작성하신 로직에 맞게 조정 필요
        
        self.minprob = minprob
        self.maxprob = maxprob
# 훨씬 깔끔해진 forward_layer
    def forward_layer(self, x, batch_adj, li):
        # x: [Batch, Nodes] -> [Batch, Nodes, 1]
        hs = x.unsqueeze(-1)
        
        # [핵심] 이제 그냥 함수처럼 호출 가능합니다!
        # 내부에서 conv1 -> conv2 -> conv3 -> fc 순차 실행됨
        hs = self.blocks[li](hs, batch_adj)
        
        # 확률 계산 및 샘플링 로직
        p = torch.sigmoid(hs)
        p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)
        p = p * (self.maxprob - self.minprob) + self.minprob
        u = torch.bernoulli(p)
        
        return u, p

class Condnet_model(nn.Module):
    def __init__(self, args, num_input = 28**2):
        super().__init__()
        # get args
        self.lambda_s = args.lambda_s
        self.lambda_v = args.lambda_v
        self.lambda_l2 = args.lambda_l2
        self.lambda_pg = args.lambda_pg
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.max_epochs = args.max_epochs
        self.BATCH_SIZE = args.BATCH_SIZE
        self.num_input = num_input
        self.condnet_min_prob = args.condnet_min_prob
        self.condnet_max_prob = args.condnet_max_prob
        self.compact = args.compact
        self.mlp = Mlp().to(device)
        self.gnn = Gnn(minprob = self.condnet_min_prob, maxprob = self.condnet_max_prob).to(device)
        self.mlp_surrogate = Mlp().to(device)
        # copy weights in mlp to mlp_surrogate
        # self.mlp_surrogate.load_state_dict(self.mlp.state_dict())

        self.C = nn.CrossEntropyLoss()
    #
    # def forward(self, x):
    #     # x : input
    #     # get policy
    #     u, p = self.gnn(x, adj_)
    #     # get output
    #     y, hs = self.mlp(x, cond_drop=self.compact, us=u)
    #     return y, p, hs, u


def adj(model, bidirect = True, last_layer = True, edge2itself = True):
    if last_layer:
        num_nodes = sum([layer.in_features for layer in model.layers]) + model.layers[-1].out_features
        nl = len(model.layers)
        trainable_nodes = np.concatenate(
            (np.ones(sum([layer.in_features for layer in model.layers])), np.zeros(model.layers[-1].out_features)),
            axis=0)
        # trainable_nodes => [1,1,1,......,1,0,0,0] => input layer & hidden layer 의 노드 개수 = 1의 개수, output layer 의 노드 개수 = 0의 개수
    else:
        num_nodes = sum([layer.in_features for layer in model.layers])
        nl = len(model.layers) - 1
        trainable_nodes = np.ones(num_nodes)

    adjmatrix = np.zeros((num_nodes, num_nodes), dtype=np.int16)
    current_node = 0

    for i in range(nl):
        layer = model.layers[i]
        num_current = layer.in_features
        num_next = layer.out_features

        for j in range(current_node, current_node + num_current):
            for k in range(current_node + num_current, current_node + num_current + num_next):
                adjmatrix[j, k] = 1
        # print start and end for j
        print(current_node, current_node + num_current)
        # print start and end for k
        print(current_node + num_current, current_node + num_current + num_next)
        current_node += num_current

    if bidirect:
        adjmatrix += adjmatrix.T

    if edge2itself:
        adjmatrix += np.eye(num_nodes, dtype=np.int16)
        # make sure every element that is non-zero is 1
    adjmatrix[adjmatrix != 0] = 1
    return adjmatrix, trainable_nodes


def adj_layerwise(model, hs, bidirect = True, last_layer = True, edge2itself = True):
    adjmatrix_list = []
    
    for li in range(len(hs)):
        num_nodes = [hs[ii].shape[1] for ii in range(len(hs[:li+1]))]
        num_nodes_ = sum(num_nodes)
        # adjmatrix = np.zeros((num_nodes, num_nodes), dtype=np.int16)
        if li == 0:
            adjmatrix = np.ones((num_nodes_, num_nodes_), dtype=np.int16)
        else:
            adjmatrix = np.zeros((num_nodes_, num_nodes_), dtype=np.int16)
            num_nodes__ = [0, *np.cumsum(num_nodes)]
            for count, num in enumerate(num_nodes__):
                if count == 0:
                    continue
                adjmatrix[num_nodes__[count-1]:num_nodes__[count], num_nodes__[count-1]:num_nodes__[count]] = 1
        current_node = 0
        
        for i in range(1, li+1):
            if i >= len(model.layers):
                break
            layer = model.layers[i]
            num_current = layer.in_features
            num_next = layer.out_features
            for j in range(current_node, current_node + num_current):
                for k in range(current_node + num_current, current_node + num_current + num_next):
                # for k in range(current_node, current_node + num_current):
                    adjmatrix[j, k] = 1
            # print start and end for j
            print(current_node, current_node + num_current)
            # print start and end for k
            print(current_node + num_current, current_node + num_current + num_next)
            current_node += num_current

        if bidirect:
            adjmatrix += adjmatrix.T

        # if edge2itself:
        #     adjmatrix += np.eye(num_nodes, dtype=np.int16)
            # make sure every element that is non-zero is 1
        adjmatrix[adjmatrix != 0] = 1
        adjmatrix_list.append(adjmatrix)
        
    return adjmatrix_list


def main():
    # get args
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--nlayers', type=int, default=3)
    args.add_argument('--lambda_s', type=float, default=7)
    args.add_argument('--lambda_v', type=float, default=0.2)
    args.add_argument('--lambda_l2', type=float, default=5e-4)
    args.add_argument('--lambda_pg', type=float, default=1e-3)
    args.add_argument('--max_epochs', type=int, default=200)
    args.add_argument('--tau', type=float, default=0.3)
    args.add_argument('--condnet_min_prob', type=float, default=0.01)
    args.add_argument('--condnet_max_prob', type=float, default=0.99)
    args.add_argument('--learning_rate', type=float, default=0.03)
    args.add_argument('--BATCH_SIZE', type=int, default=256)
    args.add_argument('--compact', type=bool, default=False)
    args.add_argument('--hidden-size', type=int, default=64)
    args.add_argument('--cuda', type=int, default=0)
    args.add_argument('--warmup', type=int, default=0)
    args.add_argument('--multi', type=float, default=0.99)
    args = args.parse_args()

    lambda_s = args.lambda_s
    lambda_v = args.lambda_v
    lambda_l2 = args.lambda_l2
    lambda_pg = args.lambda_pg
    tau = args.tau
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    BATCH_SIZE = args.BATCH_SIZE
    condnet_min_prob = args.condnet_min_prob
    condnet_max_prob = args.condnet_max_prob
    compact = args.compact
    num_inputs = 28**2

    mlp_model = Mlp().to(device)
    outputs, hs = mlp_model(torch.randn(1, num_inputs).to(device))
    gnn_policy = Gnn(hs = hs, minprob=condnet_min_prob, maxprob=condnet_max_prob, hidden_size=args.hidden_size).to(device)
    mlp_model = Mlp(gnn_policy=gnn_policy).to(device)

    # model = Condnet_model(args=args.parse_args())

    # mlp_surrogate = Mlp().to(device)
    # # copy weights in mlp to mlp_surrogate
    # mlp_surrogate.load_state_dict(mlp_model.state_dict())

    num_params = 0
    for param in mlp_model.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

    num_params = 0
    for param in gnn_policy.parameters():
        num_params += param.numel()
    print('Number of parameters: {}'.format(num_params))

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

    wandb.init(project="condgtest_2026",
                entity="hails",
                config=args.__dict__,
                name='condgstepwise_mlp_schedule_s=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + '_ti-'
                )

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(gnn_policy.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=lambda_l2)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return 0.0  # 처음 warmup 에포크 동안 학습률 0
        else:
            return args.learning_rate * args.multi ** (epoch - args.warmup)  # warmup 이후에는 지수적으로 증가

    policy_scheduler = torch.optim.lr_scheduler.LambdaLR(policy_optimizer, lr_lambda)

    adj_ = adj_layerwise(mlp_model, hs)
    adj_ = [torch.stack([torch.Tensor(adj_[i]) for _ in range(BATCH_SIZE)]).to(device) for i in range(len(adj_))]

    mlp_model.train()
    # run for 50 epochs
    for epoch in range(max_epochs):
        bn =0
        costs = 0
        accs = 0
        accsbf = 0
        PGs = 0
        num_iteration = 0
        taus = 0
        Ls = 0
        us = torch.zeros((1562, 1562))

        gnn_policy.train()
        mlp_model.train()
        policy_scheduler.step()

        # run for each batch
        for i, data in enumerate(train_loader, 0):


            if args.compact:
                if i>50:
                    break

            mlp_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data
            # get batch

            inputs = inputs.view(-1, num_inputs).to(device)

            # Forward Propagation
            # mlp_surrogate.eval()
            # outputs_1, hs = mlp_surrogate(inputs)
            # hs = torch.cat(tuple(hs[i] for i in range(len(hs))),
            #                dim=1)  # changing dimension to 1 for putting hs vector in gnn
            # hs = hs.detach()

            current_batch_size = inputs.shape[0]
            if current_batch_size < BATCH_SIZE:
                adj_batch = [adj_[ii][:current_batch_size] for ii in range(len(adj_))]
            else:
                adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

            # us, p = gnn_policy(hs, adj_batch)  # run gnn
            outputs, hs, uslist, pslist = mlp_model.forward_layerwise(inputs, adj_batch, cond_drop=True)

            ps = torch.cat(pslist, dim=1)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels.to(device))
            # Compute the regularization loss L

            # Lb_ = torch.norm(p.squeeze().mean(axis=0) - torch.tensor(tau).to(device), p=2)
            Lb_ = torch.pow(ps.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean()
            Le_ = torch.pow(ps.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean()

            L = c + lambda_s * (Lb_)

            Lv_ =  -torch.norm(ps.squeeze() - ps.squeeze().mean(axis=0), p=2, dim=0).mean()
            # Lv_ = (-1)* (p.squeeze().var(axis=0).mean()).mean()


            L += lambda_v * Lv_



            # Compute the policy gradient (PG) loss
            logp = torch.log(ps.squeeze()).sum(axis=1).mean()
            PG = lambda_pg * c.detach() * (-logp) + L
            # PG = lambda_pg * c * (-logp) + L

            PG.backward() # it needs to be checked [TODO]
            mlp_optimizer.step()
            policy_optimizer.step()
            
            outputs_1, _, _, _ = mlp_model.forward_layerwise(inputs, adj_batch, cond_drop=False)

            # calculate accuracy
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
            pred_1 = torch.argmax(outputs_1.to('cpu'), dim=1)
            accbf = torch.sum(pred_1 == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += c.to('cpu').item()
            accs += acc
            accsbf += accbf
            PGs += PG.to('cpu').item()
            Ls += L.to('cpu').item()


            tau_ = us.mean().detach().item()
            taus += tau_
            # wandb log training/batch
            wandb.log({'train/batch_cost': c.item(), 'train/batch_acc': acc, 'train/batch_acc_bf': accbf, 'train/batch_pg': PG.item(), 'train/batch_loss': L.item(), 'train/batch_tau': tau_, 'train/batch_Lb': Lb_, 'train/batch_Le': Le_, 'train/batch_Lv': Lv_, 'train/lr_policy': policy_optimizer.param_groups[0]['lr']})

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.10f}, PG:{:.5f}, Lb {:.3f}, Lv {:.3f}, Acc: {:.3f}, Acc: {:.3f}, Tau: {:.3f}'.format(epoch, i, c.item(), PG.item(), Lb_, Lv_, acc, accbf,tau_ ))

        # wandb log training/epoch
        wandb.log({'train/epoch_cost': costs / bn, 'train/epoch_acc': accs / bn, 'train/epoch_acc_bf': accsbf / bn, 'train/epoch_tau': taus / bn, 'train/epoch_PG': PGs/bn, 'train/epoch_PG': Ls/bn})

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        costs = 0
        accs = 0
        PGs = 0

        gnn_policy.eval()
        mlp_model.eval()
        with torch.no_grad():

            bn = 0
            costs = 0
            accs = 0
            accsbf = 0
            PGs = 0
            num_iteration = 0
            taus = 0
            Ls = 0
            us = torch.zeros((1562, 1562))

            gnn_policy.train()
            mlp_model.train()

            # run for each batch
            for i, data in enumerate(test_loader, 0):

                bn += 1
                # get batch
                inputs, labels = data
                # get batch

                inputs = inputs.view(-1, num_inputs).to(device)
                
                
                current_batch_size = inputs.shape[0]
                if current_batch_size < BATCH_SIZE:
                    adj_batch = [adj_[ii][:current_batch_size] for ii in range(len(adj_))]
                else:
                    adj_batch = adj_  # 기본적으로 설정된 BATCH_SIZE 크기의 adj_ 사용

                outputs, hs, uslist, pslist = mlp_model.forward_layerwise(inputs, adj_batch, cond_drop=True)
                ps = torch.cat(pslist, dim=1)

                # make labels one hot vector
                y_one_hot = torch.zeros(labels.shape[0], 10)
                y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

                c = C(outputs, labels.to(device))
                # Compute the regularization loss L

                Lb_ = torch.pow(ps.squeeze().mean(axis=0) - torch.tensor(tau).to(device), 2).mean()
                Le_ = torch.pow(ps.squeeze().mean(axis=1) - torch.tensor(tau).to(device), 2).mean()

                L = c + lambda_s * (Lb_)

                Lv_ =  -torch.norm(ps.squeeze() - ps.squeeze().mean(axis=0), p=2, dim=0).sum()
                L += lambda_v * Lv_

                # Compute the policy gradient (PG) loss
                logp = torch.log(ps.squeeze()).sum(axis=1).mean()
                PG = lambda_pg * c * (-logp) + L
                
                outputs_1, _, _, _ = mlp_model.forward_layerwise(inputs, adj_batch, cond_drop=False)    

                # calculate accuracy
                pred = torch.argmax(outputs.to('cpu'), dim=1)
                acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
                pred_1 = torch.argmax(outputs_1.to('cpu'), dim=1)
                accbf = torch.sum(pred_1 == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

                # addup loss and acc
                costs += c.to('cpu').item()
                accs += acc
                accsbf += accbf
                PGs += PG.to('cpu').item()
                Ls += L.to('cpu').item()

                tau_ = us.mean().detach().item()
                taus += tau_

            # wandb log training/epoch
            wandb.log({'test/epoch_cost': costs / bn, 'test/epoch_acc': accs / bn, 'test/epoch_acc_bf': accsbf / bn,
                       'test/epoch_tau': taus / bn, 'test/epoch_PG': PGs / bn, 'test/epoch_L': Ls / bn})
        # save model
        torch.save(mlp_model.state_dict(), './schedule_mlp_model_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')
        torch.save(gnn_policy.state_dict(), './schedule_gnn_policy_'+ 's=' + str(args.lambda_s) + '_v=' + str(args.lambda_v) + '_tau=' + str(args.tau) + dt_string +'.pt')

    wandb.finish()

if __name__=='__main__':
    main()