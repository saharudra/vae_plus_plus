import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from matplotlib import pyplot as plt
import random  
import argparse

from misc.utils import *

class CelebDataloader(Dataset):
    def __init__(self, params):
        super(CelebDataloader, self).__init__()
        self.params = params

def celeb_dataloader(params):
    # TODO: Complete train and val loader code
    return train_loader, val_loader