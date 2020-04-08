import os
import re
import random
import pickle
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class Vocabulary(object):
    def __init__(self):
        self.idx = 0
        self.token2id = {}
        self.id2token = {}
        self.add_word('<unk>')

    def add_word(self,word):
        if not word in self.token2id:
            self.token2id[word] = self.idx
            self.id2token[self.idx] = word
            self.idx += 1
    def keys(self):
        return self.id2token.keys()

    def __len__(self):
        return self.idx
    
    def __call__(self,word):
        if not word in self.token2id:
            return self.token2id['<unk>']
        return self.token2id[word]

def timeit(func):
    def wrapper(*args,**kargs):
        print('{} is running...'.format(func.__name__))
        start = time.time()
        ret = func(*args,**kargs)
        print('{} s to finish {}.'.format(time.time()-start,func.__name__))
        return ret
    return wrapper


@timeit
def build_bow(doc_path,bow_path='bows.pkl',voc_path='vocab.pkl',stopwords=None,no_below=10,no_above=0.3,rebuild=False):
    '''
        params:
        doc_path: path of the text file(each document per line), e.g. .data/sohu100k/sohu100k_clean_cut_lines.txt
        bow_path: path of the pickle file of bows, default stored in the same directory as text file
        vocab_path: path of the vocabulary file of bows, default stored in the same directory as text file
        return:
        bows:{'tokens':tokens,'counts':counts}, vocab
    '''
    txtDocs_path = os.path.join(os.path.split(bow_path)[:-1][0],'txtDocs.pkl')
    if not rebuild and os.path.exists(bow_path):
        bows = pickle.load(open(bow_path,'rb'))
        vocab = pickle.load(open(voc_path,'rb'))
        txtDocs = pickle.load(open(txtDocs_path,'rb'))
        print('{} documents in total. Vocabulary size:{}'.format(len(bows),len(vocab)))
        return bows,vocab,txtDocs
    vocab = Vocabulary()
    if stopwords!=None:
        stopwords = set([line.strip('\n') for line in open(stopwords,'r',encoding='utf-8').readlines()])
    with open(doc_path,'r',encoding='utf-8') as rfp:
        docs = [re.sub('\W+', ' ', line.strip('\n')).replace("_", ' ') for line in rfp]
        docs = [[w for w in line.split() if (w.strip()!='') and (not (w.strip().isdigit()))] for line in docs]
        txtDocs = docs[:]
        flat_docs = [w for doc in docs for w in doc]
        cnter = Counter(flat_docs)
        filt, L = [], len(flat_docs)
        for w,c in cnter.items():
            if (c<no_below) or (c>=L*no_above):
                filt.append(w)
        filt = set(filt)
        docs = [[w for w in doc if (w not in filt)] for doc in docs]
        if stopwords!=None:
            print('Removing stopwords ...')
            docs = [[w for w in doc if (w not in stopwords)] for doc in docs]
        for doc in docs:
            for w in doc:
                vocab.add_word(w)
        doc_ids = [[vocab(w) for w in doc] for doc in docs]
        doc_ids = [doc for doc in doc_ids if doc!=[]]
        #tokens,counts = [],[]
        bows = []
        for doc in doc_ids:
            cnter = Counter(doc)
            bows.append(list(cnter.items()))
            #tmplst = list(zip(*list(cnter.items())))
            #tokens.append(np.array(tmplst[0]))
            #counts.append(np.array(tmplst[1]))
        #tokens = np.array(tokens)
        #counts = np.array(counts)
        #bows = {'tokens':tokens,'counts':counts}
        with open(bow_path,'wb') as wfp:
            pickle.dump(bows,wfp)
        with open(txtDocs_path,'wb') as wfp:
            pickle.dump(txtDocs,wfp)
        with open(voc_path,'wb') as wfp:
            pickle.dump(vocab,wfp)
            print('{} documents in total. Vocabulary size:{}'.format(len(bows),len(vocab)))
            #print('{} documents in total. Vocabulary size:{}'.format(len(bows['tokens']),len(vocab)))
        return bows,vocab,txtDocs

class BOWDataset(Dataset):
    def __init__(self,bow_path,vocsize,normalize=False):
        super().__init__()
        self.normalize = normalize
        self.vocsize = vocsize
        self.bows = pickle.load(open(bow_path,'rb'))
        #self.tokens = bows['tokens']
        #self.counts = bows['counts']
        self.tokens = np.array([[t[0] for t in bow] for bow in self.bows])
        self.counts = np.array([[t[1] for t in bow] for bow in self.bows])

    def __len__(self):
        return len(self.bows)

    def __getitem__(self, index):
        bow = np.zeros(self.vocsize)
        tokenLst = self.tokens[index]
        bow[tokenLst] = self.counts[index]
        if self.normalize:
            bow = bow / bow.sum()
        return torch.from_numpy(bow).float()


def get_batch(taskname,use_stopwords=True,batch_size=128):
    print('Taskname:{}'.format(taskname))
    doc_path = os.path.join('data',taskname,'{}_clean_cut_lines.txt'.format(taskname))
    bow_path = os.path.join('data',taskname,'{}_bows.pkl'.format(taskname))
    voc_path = os.path.join('data',taskname,'{}_vocab.pkl'.format(taskname))
    stopwords = os.path.join('data','stopwords.txt') if use_stopwords==True else None
    bows,voc,txtDocs = build_bow(doc_path,bow_path,voc_path,stopwords)
    dataset = BOWDataset(bow_path,len(voc))
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    return dataloader,voc,txtDocs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Help to build the BOW file')
    parser.add_argument('--taskname',type=str,default='sohu100k',help='Taskname e.g sohu100k')
    parser.add_argument('--no_below',type=int,default=10,help='The lower bound of count for words to keep, e.g 10')
    parser.add_argument('--no_above',type=float,default=0.3,help='The ratio of upper bound of count for words to keep, e.g 0.3')
    args = parser.parse_args()
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    doc_path = os.path.join('data',taskname,'{}_clean_cut_lines.txt'.format(taskname))
    bow_path = os.path.join('data',taskname,'{}_bows.pkl'.format(taskname))
    voc_path = os.path.join('data',taskname,'{}_vocab.pkl'.format(taskname))
    stopwords = os.path.join('data','stopwords.txt')
    bows,voc,txtDocs = build_bow(doc_path,bow_path,voc_path,stopwords,no_below,no_above,True)
    dataset = BOWDataset(bow_path,len(voc))
    dataloader = DataLoader(dataset,batch_size=24,shuffle=True,num_workers=4)
    print(next(iter(dataloader)).shape)
