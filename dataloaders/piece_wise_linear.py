import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from matplotlib import pyplot as plt
import random  
import argparse
import math

from misc.utils import *

class PieceWiseDataloader(Dataset):
    def __init__(self, params, partition='train', transform=None):
        super(PieceWiseDataloader, self).__init__()
        self.params = params
        self.partition = partition
        self.transform = transform
        self.points = self.give_piecewise_points()

    def give_piecewise_points(self, start=0, end=15, x1=5, x2=10, slope=1, bias=0, num_samples=1000):
        """
        Hand designed sampler for getting points on a piece-wise linear semi-continous line.
        """
        section_1_x = np.linspace(start, x1, num_samples)
        section_1_y = np.array([bias] * num_samples)
        section_2_x = np.linspace(x1, x2, num_samples)
        section_2_y = [i - x1 for i in section_2_x]
        slope = [slope] * num_samples
        bias = [bias] * num_samples
        section_2_y = np.add(np.multiply(section_2_y, slope), bias)
        section_3_x = np.linspace(x2, end, num_samples)
        section_3_y = [section_2_y[-1]] * num_samples 
        points = [(section_1_x, section_1_y), (section_2_x, section_2_y),
                  (section_3_x, section_3_y)]
        return points

    def __getitem__(self, idx):
        section = math.floor(idx / self.params['piece-wise']['num_samples'])
        id = idx - section * self.params['piece-wise']['num_samples']  # idx - (section the point belongs to) (num of points in each section)
        data_x = self.points[section][0][id]
        data_y = self.points[section][1][id]
        point = [data_x, data_y]
        label = [section]
        sample = {}
        sample['point'] = point 
        sample['label'] = label
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        """
        :return: length of the training and validation sets
        """
        if self.partition == 'train':
            return self.params['piece-wise']['num_samples'] * self.params['piece-wise']['num_sections']
        elif self.partition == 'val':
            return self.params['piece-wise']['num_samples'] * self.params['piece-wise']['num_sections']


class ToTensor(object):
    def __call__(self, sample):
        if isinstance(sample['point'], list):
            sample['point'] = torch.Tensor(sample['point'])
        if isinstance(sample['point'], np.ndarray):
            sample['point'] = torch.from_numpy(sample['point'])
        if isinstance(sample['label'], list):
            sample['label'] = torch.Tensor(sample['label'])
        if isinstance(sample['label'], np.ndarray):
            sample['label'] = torch.from_numpy(sample['label'])
        return sample

def piece_wise_dataloader(params):
    trans=[ToTensor()]

    train_set = PieceWiseDataloader(params=params, partition='train', transform=transforms.Compose(trans))
    val_set = PieceWiseDataloader(params=params, partition='val', transform=transforms.Compose(trans))

    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['use_cuda']}

    train_loader = DataLoader(dataset=train_set, batch_size=params['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=val_set, batch_size=params['batch_size'], shuffle=False, **kwargs)

    return train_loader, val_loader


def piece_wise_plot(data_x, data_y):
    plt.title('Plotting piece-wise linear dataset')
    plt.scatter(data_x, data_y, marker='.', label='val_set', color='green')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/baseline_vae.yml', help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    train_loader, val_loader = piece_wise_dataloader(params)

    for idx, sample in enumerate(train_loader):
        print(sample['point'])
        data_x = sample['point'][:, 0].numpy()
        data_y = sample['point'][:, 1].numpy()
        piece_wise_plot(data_x, data_y)
        print("Only checking PieceWiseDataloader")
        import pdb; pdb.set_trace()    
