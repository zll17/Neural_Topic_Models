#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NTM_run.py
@Time    :   2020/09/30 15:52:35
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import pickle
import argparse
import time
import torch
from models import GSM
from dataset import DocDataset
from device_helper import default_device

parser = argparse.ArgumentParser('GSM topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=100,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--bkpt_continue',action='store_true',help='Resume from --ckpt (requires --ckpt)')
parser.add_argument('--use_tfidf',action='store_true',help='Use TF-IDF features for BOW input')
parser.add_argument('--rebuild',action='store_true',help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
parser.add_argument('--batch_size',type=int,default=512,help='Batch size (default=512)')
parser.add_argument('--criterion',type=str,default='cross_entropy',help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--ckpt',type=str,default=None,help='Checkpoint path')
parser.add_argument('--lang',type=str,default="zh",help='Language of the dataset')

args = parser.parse_args()

def main():
    if args.bkpt_continue and not args.ckpt:
        raise SystemExit('error: --bkpt_continue requires --ckpt')

    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    batch_size = args.batch_size
    criterion = args.criterion
    auto_adj = args.auto_adj
    ckpt = args.ckpt
    lang = args.lang

    device = default_device()
    docSet = DocDataset(taskname,lang=lang,no_below=no_below,no_above=no_above,rebuild=args.rebuild,use_tfidf=args.use_tfidf)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,lang=lang,no_below=no_below,no_above=no_above,rebuild=args.rebuild,use_tfidf=args.use_tfidf)
    
    voc_size = docSet.vocabsize
    print('voc size:',voc_size)

    if ckpt:
        checkpoint=torch.load(ckpt)
        param=checkpoint["param"]
        param.update({"device": device})
        model = GSM(**param)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion,ckpt=checkpoint)
    else:
        model = GSM(bow_dim=voc_size,n_topic=n_topic,taskname=taskname,device=device)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion)
    model.evaluate(test_data=docSet)
    save_name = f'./ckpt/GSM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    torch.save(model.vae.state_dict(),save_name)
    txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
    with open('topic_dist_gsm.txt','w',encoding='utf-8') as wfp:
        for t,e in zip(txt_lst,embeds):
            wfp.write(f'{e}:{t}\n')
    pickle.dump({'txts':txt_lst,'embeds':embeds},open('gsm_embeds.pkl','wb'))

if __name__ == "__main__":
    main()
