#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   WTM_run.py
@Time    :   2020/10/04 21:03:13
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''



import pickle
import argparse
import time
import torch
from models import WTM
from dataset import DocDataset
from device_helper import default_device

parser = argparse.ArgumentParser('WLDA topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=100,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--use_tfidf',action='store_true',help='Use TF-IDF features for BOW input')
parser.add_argument('--rebuild',action='store_true',help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
parser.add_argument('--dist',type=str,default='gmm_std',help='Prior distribution for latent vectors: (dirichlet,gmm_std,gmm_ctm,gaussian etc.)')
parser.add_argument('--batch_size',type=int,default=512,help='Batch size (default=512)')
parser.add_argument('--dropout',type=float,default=0.4,help='Dropout for WAE encoder/decoder (default: 0.4)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument(
    '--ckpt',
    type=str,
    default=None,
    help='Resume training: path to a training checkpoint (dict with param/net/optimizer/epoch).',
)
parser.add_argument('--lang',type=str,default="zh",help='Language of the dataset')

args = parser.parse_args()


def main():
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    dist = args.dist
    batch_size = args.batch_size
    dropout = args.dropout
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
        model = WTM(**param)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,ckpt=checkpoint)
    else:
        model = WTM(bow_dim=voc_size,n_topic=n_topic,device=device,dist=dist,taskname=taskname,dropout=dropout)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0)
    model.evaluate(test_data=docSet)
    ts = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_name = f'./ckpt/WTM_{taskname}_tp{n_topic}_{dist}_{ts}.ckpt'
    torch.save(
        {
            'net': model.wae.state_dict(),
            'param': {
                'bow_dim': model.bow_dim,
                'n_topic': model.n_topic,
                'taskname': taskname,
                'dist': model.dist,
                'dropout': model.dropout,
            },
        },
        save_name,
    )
    print('Saved inference-ready checkpoint to', save_name)
    txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
    with open('topic_dist_wtm.txt','w',encoding='utf-8') as wfp:
        for t,e in zip(txt_lst,embeds):
            wfp.write(f'{e}:{t}\n')
    pickle.dump({'txts':txt_lst,'embeds':embeds},open('wtm_embeds.pkl','wb'))
    
    
if __name__ == "__main__":
    main()
