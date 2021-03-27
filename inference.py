#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2021/01/17
@Author  :   NekoMt.Tai
@Version :   1.0
@Contact :  
@Desc    :   None
'''



import os
import re
import torch
import argparse
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
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--model_path',type=str,default='',help='Load model for inference from this path')
parser.add_argument('--save_dir',type=str,default='./',help='Save inference result')
parser.add_argument('--model_name',type=str,default='WTM',help='Neural Topic Model name')
parser.add_argument('--test_path',type=str,default='',help='Test set path')

args = parser.parse_args()


def main():
    global args
    no_below = args.no_below
    no_above = args.no_above
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    use_tfidf = args.use_tfidf
    model_path = args.model_path
    model_name = args.model_name
    save_dir = args.save_dir
    test_path=args.test_path

    device = torch.device('cuda')
    
    # load checkpoint
    checkpoint=torch.load(model_path)

    # load dictionary
    taskname=checkpoint["param"]["taskname"]
    cwd = os.getcwd()
    tmpDir = os.path.join(cwd,'data',taskname)
    if os.path.exists(os.path.join(tmpDir,'corpus.mm')):
        dictionary = Dictionary.load_from_text(os.path.join(tmpDir,'dict.txt'))
    else:
        raise Exception("Build corpus first")
    
    # load test dataset
    testSet = TestData(dictionary=dictionary,lang="en",txtPath=test_path,no_below=no_below,no_above=no_above,use_tfidf=use_tfidf)
    
    # load model
    param=checkpoint["param"]
    param.update({"device": device})
    Model=globals()[model_name]
    model = Model(**param)
    model.load_model(checkpoint["net"])
    
    # inference
    infer_topics=[]
    for doc in tqdm(testSet):
        if doc==[] or doc is None:
            infer_topics.append(None)
        else:
            #infer_topics.append(int(np.argmax(model.inference(doc_tokenized=doc, dictionary=dictionary))))
            infer_topics.append(model.inference(doc_tokenized=doc, dictionary=dictionary).tolist())

    # show topics
    for i,topic in enumerate(model.show_topic_words(dictionary=dictionary)):
        print("topic{}: {}".format(i,str(topic)))

    # show the first 10 results
    with open(test_path,"r")as f:
        for i in range(10):
            print(f.readline(), " + ".join(["topic{}*{}".format(j,round(w,3)) for j,w in sorted(enumerate(infer_topics[i]),key=lambda x:x[1],reverse=True)]))

    # save results
    with open(save_dir+"/inference_result_{}_{}.json".format(model_name, time.strftime("%Y-%m-%d-%H-%M", time.localtime())),"w")as f:
        json.dump(infer_topics,f)
        print("Inference result saved.")
    

if __name__ == "__main__":
    main()
