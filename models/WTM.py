#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   WTM.py
@Time    :   2020/10/06 17:13:43
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import os
import re
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .wae import WAE
import sys
sys.path.append('..')
from utils import evaluate_topic_quality, smooth_curve

class WTM:
    def __init__(self, bow_dim=10000, n_topic=20, device=None, dist='gmm_std', taskname=None, dropout=0.0):
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        self.wae = WAE(encode_dims=[bow_dim, 1024, 512, n_topic], decode_dims=[n_topic, 512, bow_dim], dropout=dropout, nonlin='relu')
        self.device = device
        self.id2token = None
        self.dist = dist
        self.dropout = dropout
        self.taskname = taskname
        if device != None:
            self.wae = self.wae.to(device)

    def train(self, train_data, batch_size=256, learning_rate=1e-3, test_data=None, num_epochs=100, is_evaluate=False, log_every=5, beta=1.0, ckpt=None):
        self.wae.train()
        self.id2token = {v: k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4, collate_fn=train_data.collate_fn)

        optimizer = torch.optim.Adam(self.wae.parameters(), lr=learning_rate)
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
                optimizer.zero_grad()

                txts, bows = data
                bows = bows.to(self.device)

                bows_recon, theta_q = self.wae(bows)
                
                theta_prior = self.wae.sample(dist=self.dist, batch_size=len(bows), ori_data=bows).to(self.device)

                logsoftmax = torch.log_softmax(bows_recon, dim=1)
                rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                #rec_loss = F.binary_cross_entropy(torch.softmax(bows_recon,dim=1),bows,reduction='sum')
                #rec_loss = F.binary_cross_entropy(bows_recon,bows,reduction='sum')
                mmd = self.wae.mmd_loss(theta_q, theta_prior, device=self.device, t=0.1)
                #mmd = self.wae.mmd_loss(hid_vecs, theta_prior, device=self.device, t=0.1)
                s = torch.sum(bows)/len(bows)
                lamb = (5.0*s*torch.log(torch.tensor(1.0 *bows.shape[-1]))/torch.log(torch.tensor(2.0)))
                mmd = mmd * lamb

                loss = rec_loss + mmd * beta

                loss.backward()
                optimizer.step()

                trainloss_lst.append(loss.item()/len(bows))
                epochloss_lst.append(loss.item()/len(bows))
                if (iter+1) % 10 == 0:
                    print(f'Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tLoss:{loss.item()/len(bows):<.7f}\tRec Loss:{rec_loss.item()/len(bows):<.7f}\tMMD:{mmd.item()/len(bows):<.7f}')
            #scheduler.step()
            if (epoch+1) % log_every == 0:
                save_name = f'./ckpt/WTM_{self.taskname}_tp{self.n_topic}_{self.dist}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{epoch+1}.ckpt'
                checkpoint = {
                    "net": self.wae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "bow_dim": self.bow_dim,
                        "n_topic": self.n_topic,
                        "taskname": self.taskname,
                        "dist": self.dist,
                        "dropout": self.dropout
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
                plt.savefig('wlda_trainloss.png')
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
        plt.savefig(f'wlda_tc_scores.png')


    def evaluate(self, test_data, calc4each=False):
        topic_words = self.show_topic_words()
        return evaluate_topic_quality(topic_words, test_data, taskname=self.taskname, calc4each=calc4each)


    def inference_by_bow(self, doc_bow):
        # doc_bow: torch.tensor [vocab_size]; optional: np.array [vocab_size]
        if isinstance(doc_bow,np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.to(self.device)
        with torch.no_grad():
            self.wae.eval()
            theta = F.softmax(self.wae.encode(doc_bow),dim=1)
            return theta.detach().cpu().numpy()


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
            self.wae.eval()
            theta = self.wae.encode(doc_bow)
            if normalize:
                theta = F.softmax(theta,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()

    def get_embed(self,train_data, num=1000):
        self.wae.eval()
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
        self.wae.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.wae.decode(idxes)  # word_dist: [n_topic, vocab.size]
            if normalize:
                word_dist = F.softmax(word_dist,dim=1)
            return word_dist.detach().cpu().numpy()

    def show_topic_words(self, topic_id=None, topK=15, dictionary=None):
        self.wae.eval()
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device)
        word_dist = self.wae.decode(idxes)
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
        self.wae.load_state_dict(model)

if __name__ == '__main__':
    model = WAE(encode_dims=[1024, 512, 256, 20],
                decode_dims=[20, 128, 768, 1024])
    model = model.cuda()
    inpt = torch.randn(234, 1024).cuda()
    out, mu, log_var = model(inpt)
    print(out.shape)
    print(mu.shape)
