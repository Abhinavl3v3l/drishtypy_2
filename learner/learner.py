import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
import drishtypy.learner.regularization as regularization
# from torch_lr_finder import LRFinder
from drishtypy.utils.lr_finder.lr_finder import LRFinder
LR = []
def train(model, device, train_loader, optimizer, scheduler, criterion, L1_loss_enable=False):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)

        if (L1_loss_enable == True):
            regloss = regularization.L1_Loss_calc(model, 0.0005)
            regloss /= len(data)  # by batch size
            loss += regloss

        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={train_loss / (batch_idx + 1):0.6f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    train_loss /= len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)  # processed #
    return np.round(acc, 2), np.round(train_loss, 6)


# function to test the model on testing dataset
def test(model, device, test_loader, criterion, L1_loss_enable=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(test_loader.dataset) # For F.nll_loss
        test_loss /= len(test_loader)  # criterion = nn.CrossEntropyLoss()

        if (L1_loss_enable == True):
            regloss = regularization.L1_Loss_calc(model, 0.0005)
            regloss /= len(test_loader.dataset)  # by batch size which is here total test dataset size
            test_loss += regloss

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct / len(test_loader.dataset)
    return np.round(acc, 2), test_loss


# training the model epoc wise
def build_model(model, device, trainloader, testloader, epochs, L1_loss_flag=False, L2_penalty_val=0,lr  = 0.03):
    #     criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # lr = get_lr(model, trainloader, device="cuda")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95, nesterov=True, weight_decay=L2_penalty_val)


    #     scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(trainloader),
    #                                                 epochs=epochs)


    epochs = epochs
    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * epochs
    pct_start = 5/epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = lr,
                                                    total_steps=None,
                                                    epochs=epochs,
                                                    steps_per_epoch=len(trainloader),
                                                    pct_start=pct_start,
                                                    anneal_strategy='linear',
                                                    div_factor=1000,
                                                    final_div_factor = 1)
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        print("EPOCH:", epoch)
        acc, loss = train(model, device, trainloader, optimizer, scheduler,criterion, L1_loss_flag)
        print('LR:', scheduler.get_lr())
        train_acc.append(acc)
        train_losses.append(loss)
        LR.append(optimizer.param_groups[0]['lr'])
        # scheduler.step()

        acc, loss = test(model, device, testloader, criterion, L1_loss_flag)
        test_acc.append(acc)
        test_losses.append(loss)

    return train_acc, train_losses, test_acc, test_losses


# to get model summary
def model_summary(model, device, input_size):
    model = model.to(device)
    summary(model, input_size=input_size)
    return


# to get test accuracy
def get_test_accuracy(model, device, testloader):
    # model.eval()
    correct = 0
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(testloader.dataset)
    print('\nAccuracy of the network on the {} test images: {:.2f}%\n'.format(len(testloader.dataset), acc))

    return


def get_test_accuracy_cifar10(model, device, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100. * correct / total
    print('Accuracy of the network on the %d test images: %0.2f %%' % (total, acc))
    return


# to get test accuracy for each classes on test dataset
def class_based_accuracy(model, device, classes, testloader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return

def get_LR(model,  train_loader,device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.95, nesterov=True)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=8, num_iter=200, step_mode="exp")
    lr_finder.plot()
    loss = lr_finder.history['loss']
    lr = lr_finder.history['lr']
    lr = lr[loss.index(min(loss))]
    print('Using Learning Rate : ', lr)
    lr_finder.reset()

    # for index in range(len(lr_finder.history['loss'])):
    #     item = lr_finder.history['loss'][index]
    #     if item == lr_finder.best_loss:
    #         min_val_index = index
    # #     print(f"{min_val_index}") # Comments if Verbos
    # # lr_finder.plot(show_lr=lr_finder.history['lr'][75])
    # # lr_finder.plot(show_lr=lr_finder.history['lr'][min_val_index])
    #
    #
    # val_index = 75
    # mid_val_index = math.floor((val_index + min_val_index) / 2)
    # # show_lr=[{'data': lr_finder.history['lr'][val_index], 'linestyle': 'dashed'}, {'data': lr_finder.history['lr'][mid_val_index], 'linestyle': 'solid'}, {'data': lr_finder.history['lr'][min_val_index], 'linestyle': 'dashed'}]
    # # print(show_lr)
    # # IF verbose
    #
    # best_lr = lr_finder.history['lr'][mid_val_index]
    # print(f"LR to be used: {best_lr}")

    return lr

#
# def get_lr_opt(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']