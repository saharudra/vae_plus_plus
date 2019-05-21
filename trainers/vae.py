import torch 
import torch.nn as nn 
import numpy as np 

from misc.utils import *

class VAETrainer(nn.Module):
    def __init__(self, params, model, train_loader, val_loader, logger):
        super(VAETrainer, self).__init__()
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.train_iteration = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

    def train(self):
        for epoch in range(self.params['max_epochs']):
            with trange(len(self.train_loader)) as t:
                self.model.train()
                for idx, sample in enumerate(self.train_loader):
                    loss_dict = {}
                    img, label = sample

                    if self.params['use_cuda']:
                        img = img.cuda()
                    
                    self.optimizer.zero_grad()
                    vae_loss, recon_loss, kl_loss = self.model.calculate_losses(img)
                    vae_loss.backward()

                    loss_dict = info_dict('vae_loss', vae_loss.item(), loss_dict)
                    loss_dict = info_dict('recon_loss', recon_loss.item(), loss_dict)
                    loss_dict = info_dict('kl_loss', kl_loss.item(), loss_dict)

                    for tag, value in loss_dict.items():
                        self.logger.scalar_summary(tag, value, iteration)
                    
                    self.train_iteration += 1

                    t.set_postfix(loss_dict)
                    t.update()

            if epoch % self.params['loggin_interval'] == 0:
                self.val()

        return loss_dict['vae_loss'], self.train_iteration


    def val(self):
        pass 