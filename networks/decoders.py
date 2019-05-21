import torch
import torch.nn as nn 
import numpy as np 

import argparse
from misc.utils import *

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

    z = torch.randn(64, 2)
    decoder = PieceWiseDecoder(params)
    mu = decoder(z)
    print(mu.size())