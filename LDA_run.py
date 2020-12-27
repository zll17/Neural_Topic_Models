#!/usr/bin/env python
# coding: utf-8

import os
import re
import gensim
import pickle
import argparse
import logging
import time
from utils import *
from gensim.models import LdaModel,TfidfModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from dataset import DocDataset
from multiprocessing import cpu_count


parser = argparse.ArgumentParser('LDA topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.3,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_iters',type=int,default=100,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--rebuild',type=bool,default=False,help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default True)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')

args = parser.parse_args()

def main():
    global args
    
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_iters = args.num_iters
    n_topic = args.n_topic
    n_cpu = cpu_count()-2 if cpu_count()>2 else 2
    bkpt_continue = args.bkpt_continue
    use_tfidf = args.use_tfidf
    rebuild = args.rebuild
    auto_adj = args.auto_adj

    docSet = DocDataset(taskname,no_below=no_below,no_above=no_above,rebuild=rebuild)
    if auto_adj:
        no_above = docSet.topk_dfs(topk=20)
        docSet = DocDataset(taskname,no_below=no_below,no_above=no_above,rebuild=rebuild,use_tfidf=False)
    
    model_name = 'LDA'
    msg = 'bow' if not use_tfidf else 'tfidf'
    run_name= '{}_K{}_{}_{}'.format(model_name,n_topic,taskname,msg)
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    loghandler = [logging.FileHandler(filename=f'logs/{run_name}.log',encoding="utf-8")]
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',handlers=loghandler)
    logger = logging.getLogger(__name__)


    if bkpt_continue:
        print('loading model ckpt ...')
        lda_model = gensim.models.ldamodel.LdaModel.load('ckpt/{}.model'.format(run_name))


    # Training
    print('Start Training ...')

    if use_tfidf:
        tfidf = TfidfModel(docSet.bows)
        corpus_tfidf = tfidf[docSet.bows]
        #lda_model = LdaMulticore(list(corpus_tfidf),num_topics=n_topic,id2word=docSet.dictionary,alpha='asymmetric',passes=num_iters,workers=n_cpu,minimum_probability=0.0)
        lda_model = LdaModel(list(corpus_tfidf),num_topics=n_topic,id2word=docSet.dictionary,alpha='asymmetric',passes=num_iters)
    else:
        #lda_model = LdaMulticore(list(docSet.bows),num_topics=n_topic,id2word=docSet.dictionary,alpha='asymmetric',passes=num_iters,workers=n_cpu)
        lda_model = LdaModel(list(docSet.bows),num_topics=n_topic,id2word=docSet.dictionary,alpha='asymmetric',passes=num_iters)

    save_name = f'./ckpt/LDA_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'
    lda_model.save(save_name)


    # Evaluation
    print('Evaluation ...')
    topic_words = get_topic_words(model=lda_model,n_topic=n_topic,topn=15,vocab=docSet.dictionary)


    (cv_score, w2v_score, c_uci_score, c_npmi_score),_ = calc_topic_coherence(topic_words,docs=docSet.docs,dictionary=docSet.dictionary)

    topic_diversity = calc_topic_diversity(topic_words)

    result_dict = {'cv':cv_score,'w2v':w2v_score,'c_uci':c_uci_score,'c_npmi':c_npmi_score}
    logger.info('Topics:')

    for idx,words in enumerate(topic_words):
        logger.info(f'##{idx:>3d}:{words}')
        print(f'##{idx:>3d}:{words}')

    for measure,score in result_dict.items():
        logger.info(f'{measure} score: {score}')
        print(f'{measure} score: {score}')

    logger.info(f'topic diversity: {topic_diversity}')
    print(f'topic diversity: {topic_diversity}')

if __name__ == '__main__':
    main()
