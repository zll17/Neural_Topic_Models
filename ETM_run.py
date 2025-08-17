#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ETM_run.py
@Time    :   2020/09/30 15:59:22
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import os
import pickle
import argparse
import time
import numpy as np
import torch
from models import ETM
from dataset import DocDataset
from device_helper import default_device

parser = argparse.ArgumentParser('ETM topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=100,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--use_tfidf',action='store_true',help='Use TF-IDF features for BOW input')
parser.add_argument('--no_rebuild',action='store_true',help='Use cached corpus under data/<taskname> when present (default: rebuild when needed)')
parser.add_argument('--batch_size',type=int,default=512,help='Batch size (default=512)')
parser.add_argument('--criterion',type=str,default='cross_entropy',help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
parser.add_argument('--emb_dim',type=int,default=300,help="The dimension of the latent topic vectors (default:300)")
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument(
    '--ckpt',
    type=str,
    default=None,
    help='Resume training: path to a training checkpoint (dict with param/net/optimizer/epoch).',
)
parser.add_argument('--lang',type=str,default="zh",help='Language of the dataset')

args = parser.parse_args()

def _keyedvector_vocab_keys(kv):
    """gensim 4.x uses key_to_index; older versions use .vocab."""
    if hasattr(kv, 'key_to_index'):
        return list(kv.key_to_index.keys())
    return list(kv.vocab.keys())


def main():
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_epochs = args.num_epochs
    n_topic = args.n_topic
    rebuild = not args.no_rebuild
    batch_size = args.batch_size
    criterion = args.criterion
    emb_dim = args.emb_dim
    auto_adj = args.auto_adj
    ckpt = args.ckpt
    lang = args.lang

    device = default_device()
    docSet = DocDataset(taskname,lang=lang,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=args.use_tfidf)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,lang=lang,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=args.use_tfidf)
    
    voc_size = docSet.vocabsize
    print('voc size:',voc_size)

    if ckpt:
        checkpoint=torch.load(ckpt)
        param=checkpoint["param"]
        param.update({"device": device})
        model = ETM(**param)
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion,ckpt=checkpoint)
    else:
        model = ETM(bow_dim=voc_size,n_topic=n_topic,taskname=taskname,device=device,emb_dim=emb_dim) #TBD_fc1
        model.train(train_data=docSet,batch_size=batch_size,test_data=docSet,num_epochs=num_epochs,log_every=10,beta=1.0,criterion=criterion)
    model.evaluate(test_data=docSet)
    # Training loop in models/ETM.py also saves periodic checkpoints; this final file matches
    # inference.py expectation: dict with "param" + "net" (not a bare state_dict).
    ts = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_name = f'./ckpt/ETM_{taskname}_tp{n_topic}_{ts}.ckpt'
    torch.save(
        {
            'net': model.vae.state_dict(),
            'param': {
                'bow_dim': model.bow_dim,
                'n_topic': model.n_topic,
                'taskname': taskname,
                'emb_dim': model.emb_dim,
            },
        },
        save_name,
    )
    topic_vecs = model.vae.alpha.weight.detach().cpu().numpy()
    word_vecs = model.vae.rho.weight.detach().cpu().numpy()
    print('topic_vecs.shape:',topic_vecs.shape)
    print('word_vecs.shape:',word_vecs.shape)
    vocab = np.array([t[0] for t in sorted(list(docSet.dictionary.token2id.items()),key=lambda x: x[1])]).reshape(-1,1)
    topic_ids = np.array([f'TP{i}' for i in range(n_topic)]).reshape(-1,1)
    word_vecs = np.concatenate([vocab,word_vecs],axis=1)
    topic_vecs = np.concatenate([topic_ids,topic_vecs],axis=1)
    #save_name_tp = f'./ckpt/TpVec_ETM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.emb'
    save_name_wd = f'./ckpt/WdVec_ETM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.emb'
    n_instances = word_vecs.shape[0]+topic_vecs.shape[0]
    with open(save_name_wd,'w',encoding='utf-8') as wfp:
        wfp.write(f'{n_instances} {emb_dim}\n')
        wfp.write('\n'.join([' '.join(e) for e in word_vecs]+[' '.join(e) for e in topic_vecs]))
    from gensim.models import KeyedVectors
    w2v = KeyedVectors.load_word2vec_format(save_name_wd,binary=False)
    w2v_out = os.path.splitext(save_name)[0] + '.w2v'
    w2v.save(w2v_out)
    print(_keyedvector_vocab_keys(w2v))
    #w2v.most_similar('你好')
    for i in range(n_topic):
        print(f'Most similar to Topic {i}')
        print(w2v.most_similar(f'TP{i}'))
    txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
    with open('topic_dist_etm.txt','w',encoding='utf-8') as wfp:
        for t,e in zip(txt_lst,embeds):
            wfp.write(f'{e}:{t}\n')
    pickle.dump({'txts':txt_lst,'embeds':embeds},open('etm_embeds.pkl','wb'))

if __name__ == "__main__":
    main()
