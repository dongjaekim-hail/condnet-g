import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
# https://github.com/siihwanpark/FBS/blob/master/dataset.py

def get_loader(batch_size, num_workers):
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5])
    # ])
    #
    # train_data = torchvision.datasets.CIFAR10(
    #     'data', train=True, transform=transform, download=True)
    # test_data = torchvision.datasets.CIFAR10(
    #     'data', train=False, transform=transform, download=True)
    #
    # train_loader = DataLoader(train_data, batch_size=batch_size,
    #                           shuffle=True, num_workers=num_workers, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size,
    #                          shuffle=False, num_workers=num_workers, pin_memory=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.ImageNet('../data/imagenet', split='train', transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.ImageNet('../data/imagenet', split='val', transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader