import torch
from drishtypy.data.does_augmentation import get_data_transform
from drishtypy.data.get_dataset import get_dataset


def get_dataloader(batch_size, num_workers, cuda, path):
    print("Running over Cuda !! ", cuda)
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(
        shuffle=True, batch_size=64)

    train_transforms, test_transforms = get_data_transform(path)
    trainset, testset = get_dataset(train_transforms, test_transforms, path)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainset, testset, train_loader, test_loader