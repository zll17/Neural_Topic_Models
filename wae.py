import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt

# WAE model
class WAE(nn.Module):
    def __init__(self, bow_size=50000, h_dim=4096, z_dim=20,n_clust=5,nonlin='sigmoid'):
        super(WAE, self).__init__()
        self.fc1 = nn.Linear(bow_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, z_dim)
        self.fc5 = nn.Linear(z_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, bow_size)
        self.nonlin = {'relu':F.relu,'sigmoid':torch.sigmoid}[nonlin]
        self.z_dim = z_dim
        self.n_clust = n_clust
        
    def encode(self, x):
        h = self.nonlin(self.fc1(x))
        return self.fc2(h)

    def decode(self, theta):
        h = F.relu(self.fc5(theta))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        z = self.encode(x)
        z = F.softmax(z,dim=1)
        x_reconst = self.decode(z)
        return x_reconst,z 

    def sample(self,dist='dirichlet',batch_size=256,args=None):
        if dist=='dirichlet':
            z_true = np.random.dirichlet(np.ones(self.z_dim)*args['dirichlet_alpha'],size=batch_size)
            z_true = torch.from_numpy(z_true).float()
            return z_true
        elif dist=='gaussian':
            z_true = np.random.randn(batch_size,self.z_dim)
            z_true = torch.softmax(torch.from_numpy(z_true),dim=1).float()
            return z_true
        elif dist=='gmm':
            odes = np.eye(self.z_dim)
            ides = np.random.randint(low=0,high=self.z_dim,size=batch_size)
            mus = odes[ides]
            sigmas = np.ones((batch_size,self.z_dim))*0.2
            z_true = np.random.normal(mus,sigmas)
            z_true = F.softmax(torch.from_numpy(z_true).float(),dim=1)
            return z_true
        else:
            pass


    def mmd_loss(self, x,y,device,t=0.1,kernel='diffusion'):
        '''
        computes the mmd loss with information diffusion kernel
        :param x: batch_size x latent dimension
        :param y:
        :param t:
        :return:
        '''
        eps = 1e-6
        n,d = x.shape
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i+1, n):
                    sum_xx = sum_xx + torch.norm(x[i]-x[j],p=1).to(device)
            sum_xx = sum_xx / (n * (n-1))
    
            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i+1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i]-y[j],p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))
    
            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i]-y[j],p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())
    
            def diffusion_kernel(a, tmpt, dim):
                # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
                return torch.exp(-torch.acos(a).pow(2)) / tmpt
    
            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy
