"""An implementation of VaDE(https://arxiv.org/pdf/1611.05148.pdf).
"""
import math

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _reparameterize(mu, logvar):
    """Reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class VaDE(torch.nn.Module):
    """Variational Deep Embedding(VaDE).

    Args:
        n_classes (int): Number of clusters.
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, n_classes, data_dim, latent_dim):
        super(VaDE, self).__init__()

        self._pi = Parameter(torch.zeros(n_classes))
        self.mu = Parameter(torch.randn(n_classes, latent_dim))
        self.logvar = Parameter(torch.randn(n_classes, latent_dim))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU()
        )
        
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)
        self.encoder_logvar = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, data_dim),
            torch.nn.Sigmoid()
        )
       
    @property
    def weights(self):
        return torch.softmax(self._pi, dim=0)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = _reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def classify(self, x, n_samples=8):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = torch.stack(
                [_reparameterize(mu, logvar) for _ in range(n_samples)], dim=1)
            z = z.unsqueeze(2)
            h = z - self.mu
            h = torch.exp(-0.5 * torch.sum(h * h / self.logvar.exp(), dim=3))
            # Same as `torch.sqrt(torch.prod(self.logvar.exp(), dim=1))`
            h = h / torch.sum(0.5 * self.logvar, dim=1).exp()
            p_z_given_c = h / (2 * math.pi)
            p_z_c = p_z_given_c * self.weights
            y = p_z_c / torch.sum(p_z_c, dim=2, keepdim=True)
            y = torch.sum(y, dim=1)
            pred = torch.argmax(y, dim=1)
        return pred


def lossfun(model, x, recon_x, mu, logvar):
    batch_size = x.size(0)
    print('model._pi:',model._pi)
            
    # Compute gamma ( q(c|x) )
    z = _reparameterize(mu, logvar).unsqueeze(1)
    print('z:',z)
    h = z - model.mu
    print('h1:',h)
    h = torch.exp(-0.5 * torch.sum((h * h / model.logvar.exp()), dim=2))
    print('h2:',h)
    # Same as `torch.sqrt(torch.prod(model.logvar.exp(), dim=1))`
    h = h / torch.sum(0.5 * model.logvar, dim=1).exp()
    print('h3:',h)
    p_z_given_c = h / (2 * math.pi)
    print('pz|c:',p_z_given_c)
    p_z_c = p_z_given_c * model.weights
    print('p_z_c:',p_z_c)
    gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
    print('gamma:',gamma)

    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
    print('h4:',h)
    h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
    print('h:',h)
    #loss = F.binary_cross_entropy(recon_x, x, reduction='sum') 
    loss = 0.5 * torch.sum(gamma * h) \
        - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
        + torch.sum(gamma * torch.log(gamma + 1e-9)) \
        - 0.5 * torch.sum(1 + logvar)
    loss = loss / batch_size
    return loss


class AutoEncoderForPretrain(torch.nn.Module):
    """Auto-Encoder for pretraining VaDE.

    Args:
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, data_dim, latent_dim):
        super(AutoEncoderForPretrain, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            torch.nn.ReLU()
        )
        self.encoder_mu = torch.nn.Linear(2048, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, data_dim),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_mu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x
