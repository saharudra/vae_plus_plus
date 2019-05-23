import torch
import torch.nn as nn 
import numpy as np 

import argparse
from misc.utils import *

class ConvEncoder28x28(nn.Module):
    def __init__(self, params):
        super(ConvEncoder28x28, self).__init__()
        self.params = params
        self.encoder_params = params['mnist-encoder']

        self.encoder()

    def encoder(self):
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.encoder_params['ic'], self.encoder_params['nef'], 3, 2, 1),
            nn.BatchNorm2d(self.encoder_params['nef']),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(self.encoder_params['nef'], self.encoder_params['nef'] * 2, 3, 2, 1),
            nn.BatchNorm2d(self.encoder_params['nef'] * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(self.encoder_params['nef'] * 2, self.encoder_params['nef'] * 4, 3, 2, 1),
            nn.BatchNorm2d(self.encoder_params['nef'] * 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(self.encoder_params['nef'] * 4, self.encoder_params['nef'] * 8, 3, 2, 1),
            nn.BatchNorm2d(self.encoder_params['nef'] * 8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.linear_mu = nn.Sequential(
            nn.Linear(self.encoder_params['nef'] * 32, self.encoder_params['hidden']),
            nn.BatchNorm1d(self.encoder_params['hidden']),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.encoder_params['hidden'], self.encoder_params['zdim'])
        )

        self.linear_logvar = nn.Sequential(
            nn.Linear(self.encoder_params['nef'] * 32, self.encoder_params['hidden']),
            nn.BatchNorm1d(self.encoder_params['hidden']),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.encoder_params['hidden'], self.encoder_params['zdim'])
        )

    def forward(self, img):
        conv_feature_map = self.conv_encoder(img)
        flattened_feature_map = conv_feature_map.view(img.size(0), -1)
        mu, logvar = self.linear_mu(flattened_feature_map), self.linear_logvar(flattened_feature_map)
        return mu, logvar


class PieceWiseEncoder(nn.Module):
    def __init__(self, params):
        super(PieceWiseEncoder, self).__init__()
        self.params = params
        self.encoder_params = self.params['piece-wise-encoder']

        self.encoder()

    def encoder(self):
        self.linear_encoder = nn.Sequential(
            nn.Linear(self.encoder_params['ic'], self.encoder_params['hidden']),
            nn.BatchNorm1d(self.encoder_params['hidden']),
            nn.LeakyReLU(inplace=True)
        )

        self.mu = nn.Linear(self.encoder_params['hidden'], self.encoder_params['zdim'])
        self.logvar = nn.Linear(self.encoder_params['hidden'], self.encoder_params['zdim'])

    def forward(self, point):
        linear_features = self.linear_encoder(point)
        mu, logvar = self.mu(linear_features), self.logvar(linear_features)
        return mu, logvar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/baseline_vae.yml', help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    img = torch.randn(64, 1, 28, 28)
    encoder = ConvEncoder28x28(params)
    out, logvar = encoder(img)
    print(out.size())
