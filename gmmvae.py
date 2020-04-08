import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt

# VAE model
class GMMVAE(nn.Module):
    def __init__(self, bow_size=50000, h_dim=4096, z_dim=20, n_comps=5):
        super(GMMVAE, self).__init__()
        self.fc1 = nn.Linear(bow_size, h_dim)
        self.mus = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_comps)])
        self.log_vars = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_comps)])
        self.wght_comps = nn.Linear(n_comps,1)
        self.fc4 = nn.Linear(z_dim, z_dim)
        self.fc5 = nn.Linear(z_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, bow_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mus = torch.stack([fc(h) for fc in self.mus],dim=1)
        log_vars = torch.stack([fc(h) for fc in self.log_vars],dim=1)
        return mus, log_vars
    
    def reparameterize(self, mus, log_vars):
        stds = torch.exp(log_vars/2)
        epss = torch.randn_like(stds)
        zs = mus + epss * stds
        # Method 1: adopt weight sum of Zs as final z
        res = self.wght_comps(zs.transpose(1,2))
        res = res.reshape(res.shape[0],-1)
        theta = torch.softmax(res,dim=1)

        # Method 2: catenate Zs and do linear transformation to z_dim
        #zs = zs.reshape(zs.shape[0],-1)
        #res = self.fc4(zs)
        #theta = torch.softmax(res,dim=1)

        # Method 3: do softmax on each component,then do the weighted sum
        #theta = torch.softmax(self.fc4(z),dim=1)
        #theta = torch.softmax(z,dim=1)
        return theta

    def decode(self, theta):
        h = F.relu(self.fc5(theta))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        theta = self.reparameterize(mu, log_var)
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var

if __name__ == '__main__':
    data = torch.randn(3,500)
    gmmvae = GMMVAE(data.shape[1],32,24,5)
    mus,log_vars = gmmvae.encode(data)
    print('mus.shape:{},logvars.shape:{}'.format(mus.shape,log_vars.shape))
    gmmvae.reparameterize(mus,log_vars)
