#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import argparse
import logging
from utils import get_topic_words, calc_topic_coherence, calc_topic_diversity
from gensim.models import LdaModel,TfidfModel
from dataset import DocDataset


parser = argparse.ArgumentParser('LDA topic model')
parser.add_argument('--taskname',type=str,default='cnews10k',help='Taskname e.g cnews10k')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.3,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_iters',type=int,default=100,help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--bkpt_continue',action='store_true',help='Load canonical checkpoint and run lda.update() for more passes')
parser.add_argument('--use_tfidf',action='store_true',help='Train LDA on TF-IDF weighted corpus')
parser.add_argument('--rebuild',action='store_true',help='Rebuild corpus (tokenization, dict, etc.)')
parser.add_argument('--auto_adj',action='store_true',help='To adjust the no_above ratio automatically (default:rm top 20)')

args = parser.parse_args()

def main():
    taskname = args.taskname
    no_below = args.no_below
    no_above = args.no_above
    num_iters = args.num_iters
    n_topic = args.n_topic
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

    # Same path used for load (continue) and save — matches run_name (task / K / bow|tfidf).
    canonical_ckpt = os.path.join('ckpt', '{}.model'.format(run_name))

    if use_tfidf:
        tfidf = TfidfModel(docSet.bows)
        corpus_tfidf = tfidf[docSet.bows]
        train_corpus = list(corpus_tfidf)
    else:
        train_corpus = list(docSet.bows)

    if args.bkpt_continue:
        print('loading model ckpt ...')
        if not os.path.exists(canonical_ckpt):
            raise FileNotFoundError(
                'No LDA checkpoint at {}. Train once without --bkpt_continue first.'.format(canonical_ckpt)
            )
        lda_model = LdaModel.load(canonical_ckpt)
        print('Continue training (update) ...')
        lda_model.update(train_corpus, passes=num_iters)
    else:
        print('Start Training ...')
        lda_model = LdaModel(
            train_corpus,
            num_topics=n_topic,
            id2word=docSet.dictionary,
            alpha='asymmetric',
            passes=num_iters,
        )

    lda_model.save(canonical_ckpt)
    print('Saved model to', canonical_ckpt)


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
