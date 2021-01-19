#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2021/01/17
@Author  :   Yibo Liu
@Version :   1.0
@Contact :  
@Desc    :   None
'''



import os
import re
import torch
import pickle
import argparse
import logging
import time
from models import BATM, ETM, GMNTM, GSM, WTM
from utils import *
from dataset import TestData
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
import json
from gensim.corpora import Dictionary

parser = argparse.ArgumentParser('Topic model inference')
parser.add_argument('--taskname',type=str,default='cnews10k',help='taskname used for dictionary e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--dist',type=str,default='gmm_std',help='Prior distribution for latent vectors: (dirichlet,gmm_std,gmm_ctm,gaussian etc.)')
parser.add_argument('--model_path',type=str,default='',help='Load model for inference from this path')
parser.add_argument('--save_dir',type=str,default='./',help='Save inference result')
parser.add_argument('--model_name',type=str,default='WTM',help='Neural Topic Model name')
parser.add_argument('--test_path',type=str,default='',help='Test set path')

args = parser.parse_args()


def main():
    global args
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    n_topic = args.n_topic
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    use_tfidf = args.use_tfidf
    dist = args.dist
    model_path = args.model_path
    model_name = args.model_name
    save_dir = args.save_dir
    test_path=args.test_path

    device = torch.device('cuda')
    
    cwd = os.getcwd()
    tmpDir = os.path.join(cwd,'data',taskname)
    if os.path.exists(os.path.join(tmpDir,'corpus.mm')):
        dictionary = Dictionary.load_from_text(os.path.join(tmpDir,'dict.txt'))
    else:
        raise Exception("Build corpus first")
    
    testSet = TestData(dictionary=dictionary,txtPath=test_path,no_below=no_below,no_above=no_above,use_tfidf=use_tfidf)
    voc_size = testSet.vocabsize

    Model=globals()[model_name]
    model = Model(bow_dim=voc_size,n_topic=n_topic,device=device,dist=dist,taskname=taskname)
    model.load_model(model_path)
    
    topics=model.show_topic_words(dictionary=dictionary)
    for i in range(len(topics)):
        print(i, str(topics[i]))
    
    infer_topics=[]
    for doc in tqdm(testSet):
        if doc is None:
            infer_topics.append(None)
        else:
            infer_topics.append(int(np.argmax(model.inference(doc_tokenized=doc, dictionary=dictionary))))
    with open(save_dir+"/inference_result.txt","w")as f:
        json.dump(infer_topics,f)
    

if __name__ == "__main__":
    main()
