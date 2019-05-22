import torch 
import torch.nn as nn 

from models.vae import VAE
from trainers.vae import VAETrainer
from dataloaders.baseline import get_dataloaders
from misc.utils import *
from misc.logger import Logger

import pprint
import argparse
import numpy as np 
import time 
import datetime 

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/baseline_vae.yml', help='Path to config file')
opts = parser.parse_args()
params = get_config(opts.config)
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(params)

# Define dataloaders and model
train_loader, val_loader = get_dataloaders(params)
model = VAE(params)

if params['use_cuda']:
    model = model.cuda()

exp_results = params['project_root'] + '/results/' + params['exp_name'] + '_' + timestamp + '/'
exp_logs = params['project_root'] + '/logs/' + params['exp_name'] + '_' + timestamp + '/'
mkdir_p(exp_logs)
mkdir_p(exp_results)

logger = Logger(exp_logs)

vae_trainer = VAETrainer(params, model, train_loader, val_loader, logger, exp_results, exp_logs)
vae_trainer.train()

