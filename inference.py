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
from dataset import TestDocDataset
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
import json
from gensim.corpora import Dictionary


def main():
    parser = argparse.ArgumentParser('Topic model inference')
    parser.add_argument('--ckpt',type=str,help='Checkpint path')
    parser.add_argument('--model',type=str,help='Neural Topic Model name, e.g. ETM')
    parser.add_argument('--test_path',type=str,default=None,help='Test input file, default for data/[ckpt_taskname]_lines.txt')
    parser.add_argument('--lang',type=str,default="en",help='Language of the dataset, default for en')
    parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')

    args = parser.parse_args()

    # global args
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2  # unused
    ckpt = args.ckpt
    model_name = args.model
    test_path = args.test_path
    lang = args.lang
    use_tfidf = args.use_tfidf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load checkpoint
    checkpoint=torch.load(ckpt)

    taskname=checkpoint["param"]["taskname"]

    # load dictionary
    dict_path = os.path.join(os.getcwd(),"data",taskname,"dict.txt")
    dictionary = Dictionary.load_from_text(dict_path)
    
    # load test dataset
    test_path = os.path.join(os.getcwd(),'data',f'{taskname}_lines.txt') if test_path == None else test_path
    docSet = TestDocDataset(
        dictionary=dictionary,
        txtPath=test_path,
        lang=lang, 
        tokenizer=None, 
        stopwords=None, 
        use_tfidf=use_tfidf)
    
    # load model
    param=checkpoint["param"]
    param.update({"device": device})
    Model=globals()[model_name]
    model = Model(**param)
    model.load_model(checkpoint["net"])
    
    # inference
    infer_topics=[]
    for doc in tqdm(docSet):
        #infer_topics.append(int(np.argmax(model.inference(doc_tokenized=doc, dictionary=dictionary))))
        infer_topics.append(model.inference(doc_tokenized=doc, dictionary=dictionary).tolist())

    # show topics
    for i,topic in enumerate(model.show_topic_words(dictionary=docSet.dictionary)):
        print("topic{}: {}".format(i,str(topic)))

    # show the first 10 results
    with open(test_path,"r")as f:
        for i in range(10):
            print(f.readline(), " + ".join(["topic{}*{}".format(j,round(w,3)) for j,w in sorted(enumerate(infer_topics[i]),key=lambda x:x[1],reverse=True)]))

    # save results
    tmpDir = os.path.join(os.getcwd(),"inference")
    if not os.path.exists(tmpDir):
        os.mkdir(tmpDir)    
    save_path = os.path.join(tmpDir, "inference_result_{}_{}.json".format(model_name, time.strftime("%Y-%m-%d-%H-%M", time.localtime())))
    with open(save_path,"w")as f:
        json.dump(infer_topics,f)
        print("Inference result saved at %s"%save_path)
    

if __name__ == "__main__":
    main()
