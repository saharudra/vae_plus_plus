import torch 
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
from scipy.misc import logsumexp

from networks.encoders import ConvEncoder28x28
from networks.decoders import ConvDecoder28x28

class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params
        self.enc = ConvEncoder28x28(params)  # TODO: Define the encoder and the dec to be used in the param file
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
        if self.params['loss_type'] == 'bce':
            recons_loss = F.binary_cross_entropy(recons_img.view(-1, self.params['img_h'] * self.params['img_w'] * 1), 
                                                img.view(-1, self.params['img_h'] * self.params['img_w'] * 1),
                                                size_average=False)
        elif self.params['loss_type'] == 'bce_logits':
            recons_loss = F.binary_cross_entropy_with_logits(recons_img.view(-1, self.params['img_h'] * self.params['img_w'] * 3), 
                                                            img.view(-1, self.params['img_h'] * self.params['img_w'] * 3))
        elif self.params['loss_type'] == 'l1':
            recons_loss = torch.mean(torch.abs(recons_img - img))
        elif self.params['loss_type'] == 'l2':
            loss_criterion = nn.MSELoss()
            recons_loss = loss_criterion(recons_img, img)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recons_loss + self.params['kl_beta'] * kl_loss

        return loss, recons_loss, kl_loss

    def calculate_likelihood(self, img, S=5000, MB=100):
        N_test = img.size(0)

        likelihood_val = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB
        
        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = img[j].unsqueeze(0)

            a = []
            for r in range(0, int(R)):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                a_tmp, _, _ = self.calculate_loss(x)

                a.append( -a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_val.append(likelihood_x - np.log(len(a)))

        likelihood_val = np.array(likelihood_val)

        # plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_val)

    def calculate_log_likelihood(self, img, S=5000, MB=100):
        N_test = img.size(0)
        likelihood_val = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            x_single = img[j].unsqueeze(0)

            a = []

            for r in range(0, int(R)):
                a_tmp, _, _ = self.calculate_losses(x_single)
                a.append(-a_tmp.cpu().data.numpy())

            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp(a)
            likelihood_val.append(likelihood_x - np.log(len(a)))
        
        likelihood_val = np.array(likelihood_val)

        return -np.mean(likelihood_val)

    def reconstruct(self, x):
        x_mu, _, _, _ = self.forward(x)
        return x_mu

    def generate(self):
        z_sample = torch.FloatTensor(self.params['batch_size'].normal_())
        if self.params['use_cuda']:
            z_sample = z_sample.cuda()
        
        gen_x = self.dec(z_sample)
        return gen_x

    def forward(self, x):
        z_mu, z_logvar = self.enc(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.dec(z)
        return x_mu, z, z_mu, z_logvar
