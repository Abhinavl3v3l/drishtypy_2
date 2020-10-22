# Check if cuda is available
import torch


def is_cuda():
    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda available')
    else:
        print('cuda unavailable')
    return cuda


def get_device():
    device = torch.device("cuda:0" if is_cuda() else "cpu")
    print('Device set to : ', device)
    return device
