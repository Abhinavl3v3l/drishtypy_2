from torchvision import transforms
import drishtypy.data.get_dataset
import albumentations as A
from drishtypy.data.get_dataset import get_dataset
import numpy as np


def find_stats(path):
    mean = []
    stdev = []

    # Transform to Tensor
    data_transforms = A.Compose([transforms.ToTensor()])
    trainset, testset = get_dataset(data_transforms, data_transforms, path)
    data = np.concatenate([trainset.data, testset.data], axis=0, out=None)
    data = data.astype(np.float32) / 255
    for i in range(data.shape[3]):
        tmp = data[:, :, :, i].ravel()
        print('mean', tmp.mean())
        print('standard dev', tmp.std())
        mean.append(tmp.mean())
        #         mean = [i*255 for i in mean]
        stdev.append(tmp.std())
    return mean, stdev
