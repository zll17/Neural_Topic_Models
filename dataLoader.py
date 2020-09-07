#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import gensim
import pickle
import torch
from tqdm import tqdm
from tokenization import Tokenizer
from torch.utils.data import Dataset,DataLoader
from collections import Counter
from gensim.corpora import Dictionary


class DocDataset(Dataset):
    def __init__(self,taskname,txtPath=None,tokenizer=None,stopwords=None,no_below=5,no_above=0.3,hasLable=False,rebuild=True):
        txtPath = os.path.join('data',f'{taskname}_lines.txt') if txtPath==None else txtPath
        tmpDir = os.path.join('data',taskname)
        self.txtLines = [line.strip('\n') for line in open(txtPath,'r',encoding='utf-8')]
        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)
        if not rebuild and os.path.exits(os.path.join(tmpDir,'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmpDir,'corpus.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmpDir,'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmpDir,'docs.pkl'),'rb'))
        else:
            if stopwords==None:
                stopwords = set([l.strip('\t').strip() for l in open(os.path.join('data','stopwords.txt'),'r',encoding='utf-8')])
            # self.txtLines is the list of string, without any preprocessing.
            # self.texts is the list of list of tokens.
            print('Tokenizing ...')
            tokenizer = Tokenizer if tokenizer==None else tokenizer
            self.docs = [tokenizer(txt,stopwords) for txt in tqdm(self.txtLines)]
            self.docs = [line for line in self.docs if line!=[]]
            # build dictionary
            self.dictionary = Dictionary(self.docs)
            self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)  # use Dictionary to remove un-relevant tokens
            self.dictionary.compactify()
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()} # because id2token is empty be default, it is a bug.
            # convert to BOW representation
            self.bows, _docs = [],[]
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if _bow!=[]:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
            self.docs = _docs
            # serialize the dictionary
            gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir,'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmpDir,'dict.txt'))
            pickle.dump(self.docs,open(os.path.join(tmpDir,'docs.pkl'),'wb'))
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')
        
    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return self.numDocs
    
    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)


if __name__ == '__main__':
    docSet = DocDataset('cnews10k')
    dataloader = DataLoader(docSet,batch_size=64,shuffle=True,num_workers=4,collate_fn=docSet.collate_fn)
    print('docSet.docs[10]:',docSet.docs[10])
    print(next(iter(dataloader)))
