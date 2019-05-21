import torch
import torch.nn as nn 
import numpy as np 

import argparse
from misc.utils import *

class ConvDecoder28x28(nn.Module):
    def __init__(self, params):
        super(ConvDecoder28x28, self).__init__()
        self.params = params
        self.decoder_params = params['mnist-decoder']

        self.decoder()

    def decoder(self):
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.decoder_params['zdim'], self.decoder_params['hidden']),
            nn.BatchNorm1d(self.decoder_params['hidden']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.decoder_params['hidden'], self.decoder_params['hidden'] * 4),
            nn.BatchNorm1d(self.decoder_params['hidden'] * 4),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_params['ndf'] * 8, self.decoder_params['ndf'] * 4, 4, 2, 1),
            nn.BatchNorm2d(self.decoder_params['ndf'] * 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.decoder_params['ndf'] * 4, self.decoder_params['ndf'] * 2, 3, 2, 1),
            nn.BatchNorm2d(self.decoder_params['ndf'] * 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.decoder_params['ndf'] * 2, self.decoder_params['ndf'], 4, 2, 1),
            nn.BatchNorm2d(self.decoder_params['ndf']), 
            nn.LeakyReLU(inplace=True), 
            nn.ConvTranspose2d(self.decoder_params['ndf'], self.decoder_params['oc'], 4, 2, 1)
        )

    def forward(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, self.decoder_params['mid_c'], self.decoder_params['mid_h'], self.decoder_params['mid_w'])
        img_mu = self.conv_decoder(z)
        return img_mu

class PieceWiseDecoder(nn.Module):
    def __init__(self, params):
        super(PieceWiseDecoder, self).__init__()
        self.params = params
        self.decoder_params = self.params['piece-wise-decoder']

        self.decoder()

    def decoder(self):
        self.linear_decoder = nn.Sequential(
            nn.Linear(self.decoder_params['zdim'], self.decoder_params['hidden']),
            nn.BatchNorm1d(self.decoder_params['hidden']),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.decoder_params['hidden'], self.decoder_params['oc'])
        )

    def forward(self, z):
        mu = self.linear_decoder(z) 
        return mu 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/baseline_vae.yml', help='Path to config file')
    opts = parser.parse_args()
    params = get_config(opts.config)

    z = torch.randn(64, 10)
    decoder = ConvDecoder28x28(params)
    mu = decoder(z)
    print(mu.size())