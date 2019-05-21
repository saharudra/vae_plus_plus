import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

import argparse
import matplotlib.pyplot as plt

from misc.utils import *
from dataloaders.celeb import celeb_dataloader
from dataloaders.piece_wise_linear import piece_wise_dataloader

def get_dataloaders(params):
    """
    :params: Global parameter from config file
    :train_loader: Training set loader. Loads (img, label) tuple for all except piece-wise dataset.
    :val_loader: Validation set loader. Loads (img, label) tuple for all except piece-wise dataset.
    """

    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['use_cuda']}
    
    if params['dataset'] == 'mnist':
        train_loader = DataLoader(datasets.MNIST(root=params['data_root'] + '/' + params['dataset'] + '/', train=True, download=True,
                                                 transform=transforms.Compose([transforms.ToTensor()])), batch_size=params['batch_size'], 
                                                 shuffle=True, **kwargs)
        val_loader = DataLoader(datasets.MNIST(root=params['data_root'] + '/' + params['dataset'] + '/', train=False, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()])), batch_size=params['batch_size'], 
                                               shuffle=False, **kwargs)
        
    elif params['dataset'] == 'fashion-mnist':
        train_loader = DataLoader(datasets.FashionMNIST(root=params['data_root'] + '/' + params['dataset'] + '/', train=True, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor()])), batch_size=params['batch_size'], 
                                                        shuffle=True, **kwargs)
        val_loader = DataLoader(datasets.FashionMNIST(root=params['data_root'] + '/' + params['dataset'] + '/', train=False, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor()])), batch_size=params['batch_size'], 
                                                        shuffle=False, **kwargs)

    elif params['dataset'] == 'cifar10':
        train_loader = DataLoader(datasets.CIFAR10(root=params['data_root'] + '/' + params['dataset'] + '/', train=True, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor()])), batch_size=params['batch_size'], 
                                                        shuffle=True, **kwargs)
        val_loader = DataLoader(datasets.CIFAR10(root=params['data_root'] + '/' + params['dataset'] + '/', train=False, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor()])), batch_size=params['batch_size'], 
                                                        shuffle=False, **kwargs)
    
    elif params['dataset'] == 'celeb':
        train_loader, val_loader = celeb_dataloader(params)

    elif params['dataset'] == 'piece-wise':
        train_loader, val_loader = piece_wise_dataloader(params)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/baseline_vae.yml', help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    train_loader, val_loader = get_dataloaders(params)

    for idx, sample in enumerate(val_loader):
        img, label = sample
        print(img.size(), label.size())
        import pdb; pdb.set_trace()        
