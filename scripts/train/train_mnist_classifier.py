import numpy as np
import torch
import torchvision
from operator import itemgetter
import matplotlib as mpl
import math
from torch import nn
from torch.cuda import amp
from tqdm import tqdm, trange
import gc
from torch.nn import functional as F
from torch.utils.data import random_split
from copy import deepcopy

from lib.diffusion_model.resnet_classifier import ResNet
from lib.diffusion_model.cfg import load_config
from lib.diffusion_model import loader

BATCH_SIZE = 64

DATASET = 'mnist'
config = load_config(DATASET)
suffix = '{:s}_{:d}_newUnet/'.format(config.DATASET, config.n_images)
config.DEVICE = 'cuda:0'

loading_func = 'loader.load_{:s}(config, loadtest=True)'.format(config.DATASET)
testset = None
trainset, testset = eval(loading_func)


if __name__ == '__main__':
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory=False)
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=config.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  pin_memory=False)


    # Build loaders
    nb = int(len(trainset) * 0.2)
    trainset, valset = random_split(trainset, [len(trainset)-nb, nb])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1,
                                              pin_memory=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1,
                                              pin_memory=False)


    model = ResNet().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()

    param_groups = [
        {'params':model.base.parameters(),'lr':.0001},
        {'params':model.final.parameters(),'lr':.001}
    ]
    optimizer = torch.optim.Adam(param_groups)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    states = {}

    tf = torchvision.transforms.Resize(224)     # Resize the image (maybe not necessary)

    best_val_acc = -1000
    best_val_model = None
    for epoch in range(10):
        model.train(True)
        train_loss = 0.0
        train_acc = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(config.DEVICE)
            inputs = tf(inputs)
            labels = labels.to(config.DEVICE)
            # Compute loss and update parameters
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            train_acc += (labels==out).sum().item()
        print(f"Train loss {epoch+1}: {train_loss/len(trainset)}, Train Acc:{train_acc*100/len(trainset)}%")

        correct = 0
        model.train(False)
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(config.DEVICE)
                inputs = tf(inputs)
                out = model(inputs).cpu()
                out = torch.argmax(out,dim=1)
                acc = (out==labels).sum().item()
                correct += acc
        print(f"Val accuracy:{correct*100/len(valset)}%")
        if correct>best_val_acc:
            best_val_acc = correct
            best_val_model = deepcopy(model.state_dict())
        lr_scheduler.step()

    print('Finished Training')

    # Test accuracy

    correct = 0
    model.train(False)
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs = inputs.to(config.DEVICE)
            out = model(inputs).cpu()
            out = torch.argmax(out,dim=1)
            acc = (out==labels).sum().item()
            correct += acc

    print(f"Test accuracy:{correct*100/len(testset)}%")

    # Save model
    path_save_classif = '/extra/shared/groups/marinaivan/data_marina/training/mnist/classifier/'
    torch.save(model.state_dict(), path_save_classif + 'ResNet18_' + suffix)
