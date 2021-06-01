#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vade.py
@Time    :   2020/10/08 22:17:53
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture

# VaDE model
class VaDE(nn.Module):
    def __init__(self, encode_dims=[2000,1024,512,20],decode_dims=[20,1024,2000],dropout=0.0,n_clusters=10,nonlin='relu'):
        super(VaDE, self).__init__()
        self.n_clusters = n_clusters
        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })
        self.fc_pi = nn.Linear(encode_dims[-2],n_clusters)
        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1])

        self.decoder = nn.ModuleDict({
            f'dec_{i}':nn.Linear(decode_dims[i],decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])
        self.nonlin = {'relu':F.relu,'sigmoid':torch.sigmoid}[nonlin]
        
        self.pi = nn.Parameter(torch.ones(self.n_clusters,dtype=torch.float32)/self.n_clusters,requires_grad=True)
        self.mu_c = nn.Parameter(torch.zeros(self.n_clusters,self.latent_dim,dtype=torch.float32),requires_grad=True)
        self.logvar_c = nn.Parameter(torch.zeros(self.n_clusters,self.latent_dim,dtype=torch.float32),requires_grad=True)
        
        self.gmm = GaussianMixture(n_components=self.n_clusters,covariance_type='diag',max_iter=200,reg_covar=1e-5)

    def encode(self, x):
        hid = x
        for i,layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var, qc = self.fc_mu(hid), self.fc_logvar(hid), torch.softmax(self.fc_pi(hid),dim=1)
        return mu, log_var, qc

    def get_latent(self,x):
        with torch.no_grad():
            mu, log_var, qc = self.encode(x)
            return mu

    def inference(self,x):
        mu, log_var, qc = self.encode(x)
        theta = self.fc1(mu)
        return theta
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        hid = z
        for i,(_,layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        return hid  #TBD
        
    def log_pdf_gauss(self,z,mu,logvar):
        # Compute the log value of the probability of z given mu and logvar under gaussian distribution
        # i.e. log(p(z|c)) in the Equation (16) (the numerator of the last term) of the original paper
        # params: z=[batch_size * latent_dim], mu=[1 * latent_dim], logvar=[1 * latent_dim]
        # return: res=[batch_size,1], each row is the log val of the probability of a data point w.r.t the component N(mu,var)
        '''
            log p(z|c_k) &= -(J/2)log(2*pi) - (1/2)*\Sigma_{j=1}^{J} log sigma_{j}^2 - \Sigma_{j=1}^{J}\frac{(z_{j}-mu_{j})^2}{2*\sigma_{j}^{2}}
                         &=-(1/2) * \{[log2\pi,log2\pi,...,log2\pi]_{J}
                                    + [log\sigma_{1}^{2},log\sigma_{2}^{2},...,log\sigma_{J}^{2}]_{J}
                                    + [(z_{1}-mu_{1})^2/(sigma_{1}^{2}),(z_{2}-mu_{2})^2/(sigma_{2}^{2}),...,(z_{J}-mu_{J})^2/(sigma_{J}^{2})]_{J}                   
                                    \},
            where J = latent_dim
        '''
        return (-0.5*(torch.sum(np.log(2*np.pi)+logvar+(z-mu).pow(2)/torch.exp(logvar),dim=1))).view(-1,1)

    def log_pdfs_gauss(self,z,mus,logvars):
        # Compute log value of the posterion probability of z given mus and logvars under GMM hypothesis.
        # i.e. log(p(z|c)) in the Equation (16) (the second term) of the original paper.
        # params: z=[batch_size * latent_dim], mus=[n_clusters * latent_dim], logvars=[n_clusters * latent_dim]
        # return: [batch_size * n_clusters], each row is [log(p(z|c1)),log(p(z|c2)),...,log(p(z|cK))]
        log_pdfs = []
        for c in range(self.n_clusters):
            log_pdfs.append(self.log_pdf_gauss(z,mus[c:c+1,:],logvars[c:c+1,:]))
        return torch.cat(log_pdfs,dim=1)

    def gmm_kl_div(self,mus,logvars):
        # mus=[batch_size,latent_dim], logvars=[batch_size,latent_dim]
        zs = self.reparameterize(mus,logvars)
        # zs=[batch_size,latent_dim]
        mu_c = self.mu_c
        logvar_c = self.logvar_c
        # mu_c=[n_clusters,latent_dim], logvar_c=[n_clusters,latent_dim]
        delta = 1e-10
        gamma_c = torch.exp(torch.log(self.pi.unsqueeze(0))+self.log_pdfs_gauss(zs,mu_c,logvar_c))+delta
        gamma_c = gamma_c / (gamma_c.sum(dim=1).view(-1,1))
        #gamma_c = F.softmax(gamma_c*len(gamma_c)*1.2,dim=1) #amplify the discrepancy of the distribution
        # gamma_c=[batch_size,n_clusters]

        # kl_div=[batch_size,n_clusters,latent_dim], the 3 lines above are 
        # correspond to the 3 terms in the second line of Eq. (12) in the original paper, respectively.
        kl_div = 0.5 * torch.mean(torch.sum(gamma_c*torch.sum(logvar_c.unsqueeze(0)+
                        torch.exp(logvars.unsqueeze(1)-logvar_c.unsqueeze(0))+
                        (mus.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/(torch.exp(logvar_c.unsqueeze(0))),dim=2),dim=1))
        # The two sum ops are corrrespond to the Sigma wrt J and the Sigma wrt K, respectively.
        # torch.mean() is applied along the batch dimension.
        kl_div -= torch.mean(torch.sum(gamma_c*torch.log(self.pi.unsqueeze(0)/gamma_c),dim=1)) + 0.5*torch.mean(torch.sum(1+logvars,dim=1))
        # Correspond to the last two terms of Eq. (12) in the original paper.

        return kl_div

    def mus_mutual_distance(self,dist_type='cosine'):
        if dist_type=='cosine':
            norm_mu = self.mu_c/torch.norm(self.mu_c,dim=1,keepdim=True)
            cos_mu = torch.matmul(norm_mu,norm_mu.transpose(1,0))
            cos_sum_mu = torch.sum(cos_mu) # the smaller the better
        
            theta = F.softmax(self.fc1(self.mu_c),dim=1)
            cos_theta = torch.matmul(theta,theta.transpose(1,0))
            cos_sum_theta = torch.sum(cos_theta)
        
            dist = cos_sum_mu + cos_sum_theta
        else:
            mu = self.mu_c
            dist = torch.reshape(torch.sum(mu**2,dim=1),(mu.shape[0],1))+ torch.sum(mu**2,dim=1)-2*torch.matmul(mu,mu.t())
            dist = 1.0/(dist.sum() * 0.5) + 1e-12 #/ (len(mu)*(len(mu)-1)), use its inverse, then the smaller the better
        return dist

    def forward(self, x, collate_fn=None, isPretrain=False):
        mu, log_var, qc = self.encode(x)
        if isPretrain==False:
            _z = self.reparameterize(mu, log_var)
        else:
            _z = mu
        _theta = self.fc1(_z)   #TBD
        if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var, qc

if __name__ == '__main__':
    model = VaDE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024],n_clusters=10)
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var,qc = model(inpt)
    print(out.shape)
    print(mu.shape)
    print(qc.shape)