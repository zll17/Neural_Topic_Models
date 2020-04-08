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
class VAE(nn.Module):
    def __init__(self, bow_size=50000, h_dim=4096, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(bow_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, z_dim)
        self.fc5 = nn.Linear(z_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, bow_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, log_var = self.fc2(h), self.fc3(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        theta = torch.softmax(self.fc4(z),dim=1)
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
