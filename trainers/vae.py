import torch 
import torch.nn as nn 
from torchvision.utils import save_image
import numpy as np 
from tqdm import trange

from misc.utils import *

class VAETrainer(nn.Module):
    def __init__(self, params, model, train_loader, val_loader, logger, exp_result_path, exp_logs_path):
        super(VAETrainer, self).__init__()
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.exp_result_path = exp_result_path
        self.exp_logs_path = exp_logs_path
        self.train_iteration = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

    def train(self):
        for epoch in range(self.params['max_epochs']):
            with trange(len(self.train_loader)) as t:
                self.model.train()
                for idx, sample in enumerate(self.train_loader):
                    loss_dict = {}
                    img, label = sample

                    if img.size(0) != self.params['batch_size']:
                        break

                    if self.params['use_cuda']:
                        img = img.cuda()
                    
                    self.optimizer.zero_grad()
                    vae_loss, recon_loss, kl_loss = self.model.calculate_losses(img)
                    vae_loss.backward()

                    loss_dict = info_dict('vae_loss', vae_loss.item(), loss_dict)
                    loss_dict = info_dict('recon_loss', recon_loss.item(), loss_dict)
                    loss_dict = info_dict('kl_loss', kl_loss.item(), loss_dict)

                    for tag, value in loss_dict.items():
                        self.logger.scalar_summary(tag, value, self.train_iteration)
                    
                    self.train_iteration += 1
                    loss_dict = info_dict('epoch', epoch, loss_dict)
                    t.set_postfix(loss_dict)
                    t.update()

            if epoch % self.params['loggin_interval'] == 0:
                self.val(self.train_iteration, epoch)

        return loss_dict['vae_loss'], self.train_iteration


    def val(self, iteration, epoch):
        self.model.eval()
        likelihood_val = []
        for idx, sample in enumerate(self.val_loader):
            loss_dict_val = {}
            img, label = sample 

            if self.params['use_cuda']:
                img = img.cuda()
            
            vae_loss, recon_loss, kl_loss = self.model.calculate_losses(img)
            likelihood = self.model.calculate_log_likelihood(img)
            loss_dict_val = info_dict('likelihood_val', likelihood.item(), loss_dict_val)
            loss_dict_val = info_dict('vae_loss_val', vae_loss.item(), loss_dict_val)
            loss_dict_val = info_dict('recon_loss_val', recon_loss.item(), loss_dict_val)
            loss_dict_val = info_dict('kl_loss_val', kl_loss.item(), loss_dict_val)        

            for tag, value in loss_dict_val.items():
                self.logger.scalar_summary(tag, value, iteration)

            recon_img = self.model.reconstruct(img)
            gen_img = self.model.generate()
            likelihood_val.append(likelihood)
        save_image(img, self.exp_result_path + '/' + str(epoch) + '_original_img.jpg')
        save_image(recon_img, self.exp_result_path + '/' + str(epoch) + '_reconstructed_img.jpg')
        save_image(gen_img, self.exp_result_path + '/' + str(epoch) + '_generated_img.jpg')
        likelihood_val = np.mean(np.array(likelihood_val))

        print("Epoch#{}; vae_loss: {}; recon_loss: {}; kl_loss: {}; likelihood: {}".format(epoch, 
                                                                                       vae_loss, recon_loss,
                                                                                       kl_loss, likelihood_val))

            
