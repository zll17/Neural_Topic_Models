#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   BATM_run.py
@Time    :   2020/10/12 00:12:42
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import time
import torch
import argparse
from models import BATM
from dataset import DocDataset
from device_helper import default_device

parser = argparse.ArgumentParser('Bidirectional Adversarial Topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=100,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--no_tfidf',action='store_true',help='Build first pass with raw BOW (default: TF-IDF on first pass, matching original script)')
parser.add_argument('--no_rebuild',action='store_true',help='Use cached corpus under data/<taskname> when present (default: rebuild when needed)')
parser.add_argument('--batch_size',type=int,default=512,help='Batch size (default=512)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--lang',type=str,default="zh",help='Language of the dataset')

args = parser.parse_args()


def main():
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    rebuild = not args.no_rebuild
    batch_size = args.batch_size
    auto_adj = args.auto_adj
    lang = args.lang

    device = default_device()
    use_tfidf_first = not args.no_tfidf
    docSet = DocDataset(taskname,lang=lang,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=use_tfidf_first)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,lang=lang,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)
    voc_size = docSet.vocabsize

    model = BATM(bow_dim=voc_size,n_topic=n_topic,device=device, taskname=taskname)
    model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,n_critic=10)
    model.evaluate(test_data=docSet)
    save_name = f'./ckpt/BATM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    torch.save({'generator':model.generator.state_dict(),'encoder':model.encoder.state_dict(),'discriminator':model.discriminator.state_dict()},save_name)

if __name__ == "__main__":
    main()
