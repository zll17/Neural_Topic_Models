#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gan.py
@Time    :   2020/10/11 23:10:47
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Generator,self).__init__()
        self.g = nn.Sequential(
            *block(n_topic,hid_dim),
            *block(hid_dim,bow_dim,normalize=False),
            nn.Softmax(dim=1)
        )

    def inference(self,theta):
        return self.g(theta)
    
    def forward(self,theta):
        bow_f = self.g(theta)
        doc_f = torch.cat([theta,bow_f],dim=1)
        return doc_f

class Encoder(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Encoder,self).__init__()
        self.e = nn.Sequential(
            *block(bow_dim,hid_dim),
            *block(hid_dim,n_topic,normalize=False),
            nn.Softmax(dim=1)
        )

    def forward(self,bow):
        theta = self.e(bow)
        doc_r = torch.cat([theta,bow],dim=1)
        return doc_r

class Discriminator(nn.Module):
    def __init__(self,bow_dim,hid_dim,n_topic):
        super(Discriminator,self).__init__()
        self.d = nn.Sequential(
            *block(n_topic+bow_dim,hid_dim),
            *block(hid_dim,1,normalize=False)
        )

    def forward(self,reps):
        # reps=[batch_size,n_topic+bow_dim]
        score = self.d(reps)
        return score