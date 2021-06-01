#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GMNTM.py
@Time    :   2020/10/08 23:39:33
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''

import os
import re
import time
import pickle
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from .vade import VaDE
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import evaluate_topic_quality, smooth_curve


class GMNTM:
    def __init__(self, bow_dim=10000, n_topic=10, device=None, taskname=None, dropout=0.0):
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.vade = VaDE(encode_dims=[bow_dim, 500, 500, 2000, n_topic], decode_dims=[n_topic, 2000, 500, 500, bow_dim], dropout=dropout, nonlin='relu',n_clusters=n_topic)
        self.device = device
        self.id2token = None
        self.taskname = taskname
        if device != None:
            self.vade = self.vade.to(device)

    def pretrain(self,dataloader,pre_epoch=50,retrain=False,metric='cross_entropy'):
        if (not os.path.exists('.pretrain/vade_pretrain.wght')) or retrain==True:
            if not os.path.exists('.pretrain/'):
                os.mkdir('.pretrain')
            optimizer = torch.optim.Adam(itertools.chain(self.vade.encoder.parameters(),\
                self.vade.fc_mu.parameters(),\
                    self.vade.fc1.parameters(),\
                        self.vade.decoder.parameters()))

            print('Start pretraining ...')
            self.vade.train()
            for epoch in tqdm(range(pre_epoch)):
                total_loss = []
                n_instances = 0
                for data in dataloader:
                    optimizer.zero_grad()
                    txts, bows = data
                    bows = bows.to(self.device)
                    bows_recon,_mus,_log_vars = self.vade(bows,collate_fn=lambda x: F.softmax(x,dim=1),isPretrain=True)
                    #bows_recon,_mus,_log_vars = self.vade(bows,collate_fn=None,isPretrain=True)
                    if metric=='cross_entropy':
                        logsoftmax = torch.log_softmax(bows_recon,dim=1)
                        rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                        rec_loss /= len(bows)
                    elif metric=='bce_softmax':
                        rec_loss = F.binary_cross_entropy(torch.softmax(bows_recon,dim=1),bows,reduction='sum')
                    elif metric=='bce_sigmoid':
                        rec_loss = F.binary_cross_entropy(torch.sigmoid(bows_recon),bows,reduction='sum')
                    else:
                        rec_loss = nn.MSELoss()(bows_recon,bows)
                    
                    rec_loss.backward()
                    optimizer.step()
                    total_loss.append(rec_loss.item())
                    n_instances += len(bows)
                print(f'Pretrain: epoch:{epoch:03d}\taverage_loss:{sum(total_loss)/n_instances}')
            self.vade.fc_logvar.load_state_dict(self.vade.fc_mu.state_dict())
            print('Initialize GMM parameters ...')
            z_latents = torch.cat([self.vade.get_latent(bows.to(self.device)) for txts,bows in tqdm(dataloader)],dim=0).detach().cpu().numpy()
            # TBD_corvarance_type
            try:
                self.vade.gmm.fit(z_latents)

                self.vade.pi.data = torch.from_numpy(self.vade.gmm.weights_).to(self.device).float()
                self.vade.mu_c.data = torch.from_numpy(self.vade.gmm.means_).to(self.device).float()
                self.vade.logvar_c.data = torch.log(torch.from_numpy(self.vade.gmm.covariances_)).to(self.device).float()
            except:
                self.vade.mu_c.data = torch.from_numpy(np.random.dirichlet(alpha=1.0*np.ones(self.vade.n_clusters)/self.vade.n_clusters,size=(self.vade.n_clusters,self.vade.latent_dim))).float().to(self.device)
                self.vade.logvar_c.data = torch.ones(self.vade.n_clusters,self.vade.latent_dim).float().to(self.device)

            torch.save(self.vade.state_dict(),'.pretrain/vade_pretrain.wght')
            print('Store the pretrain weights at dir .pretrain/vade_pretrain.wght')

        else:
            self.vade.load_state_dict(torch.load('.pretrain/vade_pretrain.wght'))


    def train(self, train_data, batch_size=256, learning_rate=2e-3, test_data=None, num_epochs=100, is_evaluate=False, log_every=5, beta=1.0, gamma=1e7,criterion='cross_entropy', ckpt=None):
        self.vade.train()
        self.id2token = {v: k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4, collate_fn=train_data.collate_fn)

        #self.pretrain(data_loader,pre_epoch=30,retrain=True,metric='cross_entropy')
        self.pretrain(data_loader,pre_epoch=30,retrain=True,metric='bce_softmax')

        optimizer = torch.optim.Adam(self.vade.parameters(), lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        if ckpt:
            self.load_model(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        trainloss_lst, valloss_lst = [], []
        c_v_lst, c_w2v_lst, c_uci_lst, c_npmi_lst, mimno_tc_lst, td_lst = [], [], [], [], [], []
        for epoch in range(start_epoch, num_epochs):
            epochloss_lst = []
            for iter, data in enumerate(data_loader):
                #optimizer.zero_grad()

                txts, bows = data
                bows = bows.to(self.device)

                bows_recon, mus, log_vars = self.vade(bows,collate_fn=lambda x: F.softmax(x,dim=1),isPretrain=False)
                #bows_recon, mus, log_vars = self.vade(bows,collate_fn=None,isPretrain=False)

                if criterion=='cross_entropy':
                    logsoftmax = torch.log_softmax(bows_recon, dim=1)
                    rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                    rec_loss /= len(bows)
                elif criterion=='bce_softmax':
                    rec_loss = F.binary_cross_entropy(torch.softmax(bows_recon,dim=1),bows,reduction='sum')
                elif criterion=='bce_sigmoid':
                    rec_loss = F.binary_cross_entropy(torch.sigmoid(bows_recon),bows,reduction='sum')
                
                kl_div = self.vade.gmm_kl_div(mus,log_vars)
                center_mut_dists = self.vade.mus_mutual_distance()

                loss = rec_loss + kl_div * beta + center_mut_dists * gamma

                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.vade.parameters(), max_norm=20, norm_type=2)
                optimizer.step()

                trainloss_lst.append(loss.item()/len(bows))
                epochloss_lst.append(loss.item()/len(bows))

                if (iter+1) % 10 == 0:
                    print(f'Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tLoss:{loss.item()/len(bows):<.7f}\tRec Loss:{rec_loss.item()/len(bows):<.7f}\tGMM_KL_Div:{kl_div.item()/len(bows):<.7f}\tCenter_Mutual_Distance:{center_mut_dists/(len(bows)*(len(bows)-1))}')
            #scheduler.step()
            if (epoch+1) % log_every == 0:
                save_name = f'./ckpt/GMNTM_{self.taskname}_tp{self.n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_ep{epoch+1}.ckpt'
                checkpoint = {
                    "net": self.vade.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "bow_dim": self.bow_dim,
                        "n_topic": self.n_topic,
                        "taskname": self.taskname,
                    }
                }
                torch.save(checkpoint,save_name)
                print(f'Epoch {(epoch+1):>3d}\tLoss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}')
                print('\n'.join([str(lst) for lst in self.show_topic_words()]))
                print('='*30)
                smth_pts = smooth_curve(trainloss_lst)
                plt.plot(np.array(range(len(smth_pts)))*log_every, smth_pts)
                plt.xlabel('epochs')
                plt.title('Train Loss')
                plt.savefig('gmntm_trainloss.png')
                if test_data!=None:
                    c_v,c_w2v,c_uci,c_npmi,mimno_tc, td = self.evaluate(test_data,calc4each=False)
                    c_v_lst.append(c_v), c_w2v_lst.append(c_w2v), c_uci_lst.append(c_uci),c_npmi_lst.append(c_npmi), mimno_tc_lst.append(mimno_tc), td_lst.append(td)
        scrs = {'c_v':c_v_lst,'c_w2v':c_w2v_lst,'c_uci':c_uci_lst,'c_npmi':c_npmi_lst,'mimno_tc':mimno_tc_lst,'td':td_lst}
        '''
        for scr_name,scr_lst in scrs.items():
            plt.cla()
            plt.plot(np.array(range(len(scr_lst)))*log_every,scr_lst)
            plt.savefig(f'wlda_{scr_name}.png')
        '''
        plt.cla()
        for scr_name,scr_lst in scrs.items():
            if scr_name in ['c_v','c_w2v','td']:
                plt.plot(np.array(range(len(scr_lst)))*log_every,scr_lst,label=scr_name)
        plt.title('Topic Coherence')
        plt.xlabel('epochs')
        plt.legend()
        plt.savefig(f'gmntm_tc_scores.png')


    def evaluate(self, test_data, calc4each=False):
        topic_words = self.show_topic_words()
        return evaluate_topic_quality(topic_words, test_data, taskname=self.taskname, calc4each=calc4each)


    def inference_by_bow(self, doc_bow,normalize=True):
        # doc_bow: torch.tensor [vocab_size]; optional: np.array [vocab_size]
        if isinstance(doc_bow,np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.reshape(-1,self.bow_dim).to(self.device)
        with torch.no_grad():
            theta = self.vade.inference(doc_bow)
            if normalize:
                theta = F.softmax(theta,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()


    def inference(self, doc_tokenized, dictionary,normalize=True):
        doc_bow = torch.zeros(1,self.bow_dim)
        for token in doc_tokenized:
            try:
                idx = dictionary.token2id[token]
                doc_bow[0][idx] += 1.0
            except:
                print(f'{token} not in the vocabulary.')
        doc_bow = doc_bow.to(self.device)
        with torch.no_grad():
            theta = self.vade.inference(doc_bow)
            if normalize:
                theta = F.softmax(theta,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()

    def get_embed(self,train_data, num=1000):
        self.vade.eval()
        data_loader = DataLoader(train_data, batch_size=512,shuffle=False, num_workers=4, collate_fn=train_data.collate_fn)
        embed_lst = []
        txt_lst = []
        cnt = 0
        for data_batch in data_loader:
            txts, bows = data_batch
            embed = self.inference_by_bow(bows)
            embed_lst.append(embed)
            txt_lst.append(txts)
            cnt += embed.shape[0]
            if cnt>=num:
                break
        embed_lst = np.concatenate(embed_lst,axis=0)[:num]
        txt_lst = np.concatenate(txt_lst,axis=0)[:num]
        return txt_lst, embed_lst

    def get_topic_word_dist(self,normalize=True):
        self.vade.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.vade.decode(idxes)  # word_dist: [n_topic, vocab.size]
            if normalize:
                word_dist = F.softmax(word_dist,dim=1)
            return word_dist.detach().cpu().numpy()

    def show_topic_words(self, topic_id=None, topK=15, dictionary=None):
        self.vade.eval()
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device)
        #idxes = F.softmax(self.vade.fc1(self.vade.mu_c),dim=1)
        word_dist = self.vade.decode(idxes)
        word_dist = F.softmax(word_dist, dim=1)
        vals, indices = torch.topk(word_dist, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        if self.id2token==None and dictionary!=None:
            self.id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id == None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
        return topic_words

    def load_model(self, model):
        self.vade.load_state_dict(model)


if __name__ == '__main__':
    model = VaDE(encode_dims=[1024, 512, 256, 20],
                decode_dims=[20, 128, 768, 1024])
    model = model.cuda()
    inpt = torch.randn(234, 1024).cuda()
    out, mu, log_var = model(inpt)
    print(out.shape)
    print(mu.shape)
