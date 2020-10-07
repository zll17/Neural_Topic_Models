import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from .vae import VAE
import matplotlib.pyplot as plt

def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for pt in points:
      if smoothed_points:
        prev = smoothed_points[-1]
        smoothed_points.append(prev*factor+pt*(1-factor))
      else:
        smoothed_points.append(pt)
    return smoothed_points

class GSM:
    def __init__(self,bow_dim=10000,n_topic=20,device=None):
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.vae = VAE(encode_dims=[bow_dim,1024,512,n_topic],decode_dims=[n_topic,512,bow_dim],dropout=0.0)
        self.device = device
        self.id2token = None
        if device!=None:
            self.vae = self.vae.to(device)

    def train(self,train_data,batch_size=256,learning_rate=1e-3,test_data=None,num_epochs=100,is_evaluate=False,log_every=5,beta=1.0):
        self.vae.train()
        self.id2token = {v:k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=train_data.collate_fn)

        optimizer = torch.optim.Adam(self.vae.parameters(),lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        trainloss_lst, valloss_lst = [], []
        recloss_lst, klloss_lst = [],[]
        for epoch in range(num_epochs):
            epochloss_lst = []
            for iter,data in enumerate(data_loader):
                optimizer.zero_grad()

                txts,bows = data
                bows = bows.to(self.device)
                '''
                n_samples = 20
                rec_loss = torch.tensor(0.0).to(self.device)
                for i in range(n_samples):
                    bows_recon,mus,log_vars = self.vae(bows,lambda x:torch.softmax(x,dim=1))
                    
                    logsoftmax = torch.log_softmax(bows_recon,dim=1)
                    _rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                    rec_loss += _rec_loss
                rec_loss = rec_loss / n_samples
                '''
                
                bows_recon,mus,log_vars = self.vae(bows,lambda x:torch.softmax(x,dim=1))
                logsoftmax = torch.log_softmax(bows_recon,dim=1)
                rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                #rec_loss = F.binary_cross_entropy(torch.softmax(bows_recon,dim=1),bows,reduction='sum')
                #rec_loss = F.binary_cross_entropy(bows_recon,bows,reduction='sum')
                kl_div = -0.5 * torch.sum(1+log_vars-mus.pow(2)-log_vars.exp())
                
                loss = rec_loss + kl_div * beta
                
                loss.backward()
                optimizer.step()

                trainloss_lst.append(loss.item()/len(bows))
                epochloss_lst.append(loss.item()/len(bows))
                if (iter+1) % 10==0:
                    print(f'Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tLoss:{loss.item()/len(bows):<.7f}\tRec Loss:{rec_loss.item()/len(bows):<.7f}\tKL Div:{kl_div.item()/len(bows):<.7f}')
            #scheduler.step()
            if (epoch+1) % log_every==0:
                print(f'Epoch {(epoch+1):>3d}\tLoss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}')
                print('\n'.join([str(lst) for lst in self.show_topic_words()]))
                print('='*30)
                smth_pts = smooth_curve(trainloss_lst)
                plt.plot(list(range(len(smth_pts))),smth_pts)
                plt.savefig('trainloss.png')


    def evaluate(self,test_data):
        pass

    def inference(self,doc_bow):
        pass

    def get_topic_word_dist(self):
        pass

    def show_topic_words(self,topic_id=None,topK=15):
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device)
        word_dist = self.vae.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        vals,indices = torch.topk(word_dist,topK,dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        if topic_id==None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
        return topic_words





if __name__ == '__main__':
    model = VAE(encode_dims=[1024,512,256,20],decode_dims=[20,128,768,1024])
    model = model.cuda()
    inpt = torch.randn(234,1024).cuda()
    out,mu,log_var = model(inpt)
    print(out.shape)
    print(mu.shape)
