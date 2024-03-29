import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import get_loader
from dynamic_channel_pruning_fbs import CifarNet
import utils

'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fbs',
        type=utils.str2bool,
        default=False
    )
    parser.add_argument(
        '--sparsity_ratio',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--lasso_lambda',
        type=float,
        default=1e-8
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1
    )
    parser.add_argument(
        '--num_worker',
        type=int,
        default=4
    )

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='checkpoints'
    )

    parser.add_argument(
        '--pretrained',
        type=str,
        default='checkpoints/best_False_1.0.pt'
    )

    args = parser.parse_args()

    os.makedirs(args.ckpt_path, exist_ok=True)
    with open(f'{args.ckpt_path}/train_log_{args.fbs}_{args.sparsity_ratio}.tsv', 'w') as log_file:
        log_file.write(
            'epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\tbest_acc\n')
    utils.set_seed(args.seed)

    train_loader, test_loader = get_loader(args.batch_size, args.num_worker)
    model = CifarNet(fbs=args.fbs, sparsity_ratio=args.sparsity_ratio).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO: initialize current model parameters with previous model parameters
    if args.fbs:
        if args.sparsity_ratio == 1.0:
            base_state_dict = torch.load(args.pretrained)
            model_state_dict = model.state_dict()

            for k, v in model_state_dict.items():
                if 'conv' in k:
                    model_state_dict[k] = base_state_dict[k]

            model.load_state_dict(model_state_dict)

        else:
            base_state_dict = torch.load(args.pretrained)
            model_state_dict = model.state_dict()

            for k, v in model_state_dict.items():
                if 'weight' in k or 'bias' in k:
                    model_state_dict[k] = base_state_dict[k]

            model.load_state_dict(model_state_dict)

    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch: {epoch}')

        train_loss = 0
        total_num = 0
        correct_num = 0
        total_step = len(train_loader)

        model.train()
        i = 0
        for img_batch, lb_batch in train_loader:
            img_batch = img_batch.cuda()
            lb_batch = lb_batch.cuda()

            if not args.fbs:
                pred_batch = model(img_batch)
                loss = criterion(pred_batch, lb_batch)
            else:
                pred_batch, lasso = model(img_batch)
                loss = criterion(pred_batch, lb_batch) + lasso * args.lasso_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred_lb_batch = pred_batch.max(dim=1)
            total_num += lb_batch.shape[0]
            correct_num += pred_lb_batch.eq(lb_batch).sum().item()
            print(f'Step: {i}/{total_step}, Batch Loss: {loss.item()}')
            i += 1

        train_loss = train_loss / total_step
        train_acc = 100. * correct_num / total_num
        print(f'\nEpoch: {epoch}, Train Loss: {train_loss}, Train_Accuracy: {train_acc}')

        with torch.no_grad():
            test_loss = 0
            total_num = 0
            correct_num = 0
            total_step = len(test_loader)

            model.eval()
            for img_batch, lb_batch in tqdm(test_loader, total=len(test_loader)):
                img_batch = img_batch.cuda()
                lb_batch = lb_batch.cuda()

                if not args.fbs:
                    pred_batch = model(img_batch)
                    loss = criterion(pred_batch, lb_batch)
                else:
                    pred_batch, lasso = model(img_batch, True)
                    loss = criterion(pred_batch, lb_batch) + lasso * args.lasso_lambda

                test_loss += loss.item()
                _, pred_lb_batch = pred_batch.max(dim=1)
                total_num += lb_batch.shape[0]
                correct_num += pred_lb_batch.eq(lb_batch).sum().item()

            test_loss = test_loss / total_step
            test_acc = 100. * correct_num / total_num
            print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       f'{args.ckpt_path}/best_{args.fbs}_{args.sparsity_ratio}.pt')

        with open(f'{args.ckpt_path}/train_log_{args.fbs}_{args.sparsity_ratio}.tsv', 'a') as log_file:
            log_file.write(
                f'{epoch}\t{train_loss}\t{test_loss}\t{train_acc}\t{test_acc}\t{best_acc}\n')

if __name__=='__main__':
    main()