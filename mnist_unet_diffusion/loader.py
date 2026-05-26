import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from operator import itemgetter

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def subloader(torch_set, n_images, include_list=[], props=[]):
    '''
    Return a subset of n_images images from an input dataloader excluding
    the targets not in "include_list".
    '''
    indices_keep_train = []
    targets = []
    for (i, t) in enumerate(torch_set.targets):
        t = int(t)  # convert tensor to int
        if t in include_list:
            pos = include_list.index(t)
            prop_class = len(np.where(np.array(targets) == pos)[0]) / n_images
            if prop_class < props[pos] and len(targets) < n_images:
                indices_keep_train.append(i)
                targets.append(pos)
    torch_set.data = torch_set.data[indices_keep_train]
    torch_set.targets = targets
    return torch_set


def load_MNIST(config, include_list=[1,8], props=[1/2, 1/2], loadtest=False):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    include_list : list, optional
        List of indices for classes that should be kept. The default is [1,7].
    props : list, optional
        Proportion of each class. The default is [1/2, 1/2].
    loadtest : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(2),
         # transforms.Resize(224),
          # transforms.Normalize((0.5,), (0.5,))
         ])
    
    trainset = torchvision.datasets.MNIST(root=config.path_data, train=True,
                                            download=False, transform=transform)
    
    # Class indices
    classes = np.arange(0, 10)
    classes = itemgetter(*include_list)(classes)
    
    trainset = subloader(trainset, config.n_images, 
                         include_list=include_list, props=props)
    
    mean = 0.0
    std = 1.0
    if config.CENTER:
        s_t = transforms.Compose([transforms.Pad(2),])
        t_data = s_t(trainset.data)
         
        mean = torch.mean(t_data / 255.)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Pad(2),
             # transforms.Resize(224),
              transforms.Normalize((mean,), (1,))
             ])
        if config.STANDARDIZE:
            std = torch.std(t_data / 255.)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Pad(2),
                  # transforms.Resize(224),
                  transforms.Normalize((mean,), (std,))
                 ])
            
        # Relaod dataset
        trainset = torchvision.datasets.MNIST(root=config.path_data, train=True,
                                                download=False, transform=transform)
        trainset = subloader(trainset, config.n_images, include_list=include_list, props=props)
        
        if loadtest:
            testset = torchvision.datasets.MNIST(root=config.path_data, train=False,
                                                   download=False, transform=transform)
            testset = subloader(testset, config.n_images, include_list=include_list, props=props)
        else:
            testset = None
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset

