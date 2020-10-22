from torchvision import datasets, transforms

'''
 default_train_transform()
 default_test_transform()
 select_dataset
'''


def get_dataset(train_transforms, test_transforms, path):
    """

    :param train_transforms: Train Data Transformations
    :param test_transforms:  Test Data Transformations
    :param path: Dataset path
    """
    trainset = datasets.CIFAR10(path, train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10(path, train=False, download=True, transform=test_transforms)
    return trainset, testset