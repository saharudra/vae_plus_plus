import torch 
import torch.nn as nn 
import numpy as np 

from networks.encoders import ConvEncoder28x28
from networks.decoders import ConvDecoder28x28

class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params
        self.enc = ConvEncoder28x28(params)
        self.dec = ConvDecoder28x28(params)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.from_numpy(np.random.normal(0, 1, size=std.size())).float()
        if self.params['use_cuda'] == True:
            epsilon = epsilon.cuda() 
        latent = epsilon.mul(std).add_(mu)
        return latent

    def calculate_losses(self, img):
        recons_img, latent, mu, logvar = self.forward(img)
        if params['loss_type'] == 'bce':
            recons_loss = F.binary_cross_entropy(recons_img.view(-1, self.params['img_h'] * self.params['img_w'] * 1), 
                                                img.view(-1, self.params['img_h'] * self.params['img_w'] * 1),
                                                size_average=False)
        elif params['loss_type'] == 'bce_logits':
            recons_loss = F.binary_cross_entropy_with_logits(recons_img.view(-1, self.params['img_h'] * self.params['img_w'] * 3), 
                                                            img.view(-1, self.params['img_h'] * self.params['img_w'] * 3))
        elif params['loss_type'] == 'l1':
            recons_loss = torch.mean(torch.abs(recons_img - img))
        elif params['loss_type'] == 'l2':
            loss_criterion = nn.MSELoss()
            recons_loss = loss_criterion(recons_img, img)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recons_loss + kl_loss

        return loss, recons_loss, kl_loss

    def calculate_likelihood(self, img, dir, mode='val', S=5000, MB=100):
        pass

    def forward(self, x):
        z_mu, z_logvar = self.enc(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.dec(z)
        return x_mu, z, z_mu, z_logvar