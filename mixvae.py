import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt

# MIXVAE model
class MIXVAE(nn.Module):
    def __init__(self, bow_size=50000, h_dim=4096, z_dim=20,n_clust=20):
        super(MIXVAE, self).__init__()
        self.encoders = nn.ModuleList([nn.Sequential(*[
            nn.Linear(bow_size,h_dim),
            nn.ReLU()]) for i in range(n_clust)])
        self.mus = nn.ModuleList([nn.Linear(h_dim,z_dim) for i in range(n_clust)])
        self.logvars = nn.ModuleList([nn.Linear(h_dim,z_dim) for i in range(n_clust)])
        self.mid = nn.ModuleList([nn.Sequential(*[
            nn.Linear(z_dim,z_dim)]) for i in range(n_clust)])

        self.assign_p1 = nn.Linear(bow_size,h_dim)
        self.assign_p2 = nn.Linear(h_dim,n_clust)
        self.fc5 = nn.Linear(z_dim,h_dim)
        self.fc6 = nn.Linear(h_dim, bow_size)
        
    def encode(self, x):
        hs = [encoder(x) for encoder in self.encoders]
        mus = [mu(h) for mu,h in zip(self.mus,hs)]
        logvars = [logvar(h) for logvar,h in zip(self.logvars,hs)]
        return mus,logvars

    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2.0)
        eps = torch.randn_like(std)
        z = mu + eps * std
        #theta = torch.softmax(self.fc4(z),dim=1)
        #theta = torch.softmax(z,dim=1)
        return z

    def decode(self, theta):
        h = F.relu(self.fc5(theta))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        assign_p = torch.softmax(self.assign_p2(F.relu(self.assign_p1(x))),dim=1).unsqueeze(1)
        #print('assign_p.shape:',assign_p.shape)
        mus,logvars = self.encode(x)
        zs = [self.reparameterize(mu,logvar) for mu,logvar in zip(mus,logvars)]
        #print('zs[0].shape:',zs[0].shape)
        #print('len(zs):',len(zs))
        thetas = torch.cat([torch.softmax(norm(z),dim=1).unsqueeze(1) for norm,z in zip(self.mid,zs)],dim=1)
        #print('thetas.shape:',thetas.shape)
        aggt_theta = torch.matmul(assign_p,thetas).squeeze(1)
        #print('aggt_theta.shape:',aggt_theta.shape)
        x_reconst = self.decode(aggt_theta)
        #print('x_reconst.shape:',x_reconst.shape)
        return x_reconst, mus, logvars,assign_p
