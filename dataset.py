#!/usr/bin/env python
# coding: utf-8

import os
import gensim
import random
import pickle
import torch
from tokenization import *
from data_utils import *
from torch.utils.data import Dataset,DataLoader
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

class DocDataset(Dataset):
    def __init__(self,dictionary=None,bows=None,docs=None,tfidf=None):
        
        self.dictionary = dictionary
        self.bows, self.docs = bows, docs
        self.tfidf = tfidf

        self.vocabsize = 0
        self.numDocs = 0
        self.save_dir = ""

        if self.dictionary and self.bows:
            self.vocabsize = len(self.dictionary)
            self.numDocs = len(self.bows)

    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        if self.tfidf is not None:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return self.numDocs
    
    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def create(self, txt_file, lang, use_tfidf=False):
        text = file_tokenize(txt_file, lang)
        self.dictionary = build_dictionary(text)
        self.bows, self.docs = convert_to_BOW(text, self.dictionary)
        if use_tfidf:
            self.tfidf, tfidf_model = compute_tfidf(self.bows)
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)

    def save(self, save_dir="./data/tmp"):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.dictionary.save_as_text(os.path.join(save_dir,'dict.txt'))
        print("Dictionary saved to %s"%os.path.join(save_dir,'dict.txt'))
        gensim.corpora.MmCorpus.serialize(os.path.join(save_dir,'corpus.mm'), self.bows)
        print("Corpus saved to %s"%os.path.join(save_dir,'corpus.mm'))
        pickle.dump(self.docs,open(os.path.join(save_dir,'docs.pkl'),'wb'))
        print("Docs saved to %s"%os.path.join(save_dir,'docs.pkl'))
        if self.tfidf:
            gensim.corpora.MmCorpus.serialize(os.path.join(save_dir,'tfidf.mm'),self.tfidf)
            print("TF-IDF model saved to %s"%os.path.join(save_dir,'tfidf.mm'))

    def load(self, load_dir):
        print("Loading corpus from %s ..."%load_dir)
        self.bows = gensim.corpora.MmCorpus(os.path.join(load_dir,'corpus.mm'))
        self.dictionary = load_dictionary(os.path.join(load_dir,'dict.txt'))
        self.docs = pickle.load(open(os.path.join(load_dir,'docs.pkl'),'rb'))
        if os.path.exists(os.path.join(load_dir,'tfidf.mm')):
            self.tfidf = gensim.corpora.MmCorpus(os.path.join(load_dir,'tfidf.mm'))
            print("Using TF-IDF")
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        self.save_dir=load_dir

    def show_dfs_topk(self,topk=20):
        ndoc = len(self.docs)
        dfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.dfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(dfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ndoc} = {(1.0*freq/ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self,topk=20):
        ntokens = sum([v for k,v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k],fq) for k,fq in self.dictionary.cfs.items()],key=lambda x: x[1],reverse=True)[:topk]
        for i,(word,freq) in enumerate(cfs_topk):
            print(f'{i+1}:{word} --> {freq}/{ntokens} = {(1.0*freq/ntokens):>.13f}')
    
    def topk_dfs(self,topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0*dfs_topk[-1][-1]/ndoc


# class DocDataLoader:
#     def __init__(self,dataset=None,batch_size=128,shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.idxes = list(range(len(dataset)))
#         self.length = len(self.idxes)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.shuffle==True:
#             random.shuffle(self.idxes)
#         for i in range(0,self.length,self.batch_size):
#             batch_ids = self.idxes[i:i+self.batch_size]
#             batch_data = self.dataset[batch_ids]
#             yield batch_data


if __name__ == '__main__':

    txt_path = os.path.join(os.getcwd(),'data','zhdd_lines.txt')
    tmpDir = os.path.join(os.getcwd(),'data',"zhdd")
    if tmpDir and not os.path.exists(tmpDir):
        os.mkdir(tmpDir)
    use_tfidf = True

    text = file_tokenize(txt_path, "zh")
    dictionary = build_dictionary(text, tmpDir)
    bows, docs = convert_to_BOW(text, dictionary, tmpDir)
    if use_tfidf:
         tfidf, tfidf_model = compute_tfidf(bows, tmpDir)
    
    docSet = DocDataset(dictionary,bows,docs,tfidf)

    dataloader = DataLoader(docSet,batch_size=64,shuffle=True,num_workers=4,collate_fn=docSet.collate_fn)

    print('docSet.docs[10]:',docSet.docs[10])
    print(next(iter(dataloader)))
    print('The top 20 tokens in document frequency:')
    docSet.show_dfs_topk()
    print('The top 20 tokens in collections frequency:')
    input("Press any key ...")
    # docSet.show_cfs_topk()  // buggy
    input("Press any key ...")
    for doc in docSet:
        print(doc)
        break
    print(docSet.topk_dfs(20))
    print(docSet.vocabsize)