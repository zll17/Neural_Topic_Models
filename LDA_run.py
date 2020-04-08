#!/usr/bin/env python
# coding: utf-8

import os
import re
import gensim
import pickle
import argparse
from gensim import corpora,models
from gensim.models import CoherenceModel
from data import *
from utils import *
from multiprocessing import cpu_count



parser = argparse.ArgumentParser('Help to build the BOW file')
parser.add_argument('--taskname',type=str,default='sohu100k',help='Taskname e.g sohu100k')
parser.add_argument('--no_below',type=int,default=10,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.3,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs',type=int,default=100,help='Number of epochs (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic',type=int,default=20,help='Num of topics')
parser.add_argument('--bkpt_continue',type=bool,default=False,help='Whether to load the trained model and continue training.')
parser.add_argument('--use_tfidf',type=bool,default=False,help='Whether to use the tfidf feature for the BOW input')


args = parser.parse_args()

taskname = args.taskname
no_below = args.no_below
no_above = args.no_above
num_epochs = args.num_epochs
n_topic = args.n_topic
n_cpu = cpu_count()-2
bkpt_continue = args.bkpt_continue
use_tfidf = args.use_tfidf

doc_path = os.path.join('data',taskname,'{}_clean_cut_lines.txt'.format(taskname))
bow_path = os.path.join('data',taskname,'{}_bows.pkl'.format(taskname))
voc_path = os.path.join('data',taskname,'{}_vocab.pkl'.format(taskname))
stopwords = os.path.join('data','stopwords.txt')

bows,vocab,txtDocs = build_bow(doc_path,bow_path,voc_path,stopwords,no_below,no_above,False)


model_name = 'LDA'
msg = 'bow'
run_name= '{}_K{}_{}_{}'.format(model_name,n_topic,taskname,msg)


if bkpt_continue:
    lda_model = models.ldamodel.LdaModel.load('ckpt/{}.model'.format(run_name))
    logger = open('logs/{}.log'.format(run_name),'a',encoding='utf-8')
else:
    logger = open('logs/{}.log'.format(run_name),'w',encoding='utf-8')




print('Start Training ...')

if use_tfidf:
    tfidf = models.TfidfModel(bows)
    corpus_tfidf = tfidf[bows]
    lda_model = gensim.models.LdaMulticore(corpus_tfidf,num_topics=n_topic,id2word=vocab.id2token,passes=num_epochs,workers=n_cpu,minimum_probability=0.0)
else:
    lda_model = gensim.models.LdaMulticore(bows,num_topics=n_topic,id2word=vocab.id2token,passes=num_epochs,workers=n_cpu,minimum_probability=0.0)

lda_model.save('ckpt/{}.model'.format(run_name))


# Show Topics
def show_topics(model=lda_model,topn=20,n_topic=10,fix_topic=None,showWght=False):
    global vocab
    topics = []
    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]],t[1]) for t in lda_model.get_topic_terms(tp_idx)]
        else:
            return [vocab.id2token[t[0]] for t in lda_model.get_topic_terms(tp_idx)]
    if fix_topic is None:
        for i in range(n_topic):
            topics.append(show_one_tp(i))
    else:
        topics.append(show_one_tp(fix_topic))
    return topics

topics_text=show_topics()


# Evaluate the Model with Topic Coherence

# Computing the C_V score
cv_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_v')
cv_coherence_score = cv_coherence_model.get_coherence_per_topic()
cv_coherence_avg = cv_coherence_model.get_coherence()

# Computing Topic Diversity score
topic_diversity = calc_topic_diversity(topics_text)

# Computing the C_W2V score
try:
    if os.path.exists('data/{}/{}_embeddings.txt'.format(taskname,taskname)):
        keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format('data/{}/{}_embeddings.txt'.format(taskname,taskname),binary=False)
        w2v_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_w2v',keyed_vectors=keyed_vectors)
    else:
        w2v_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_w2v')
    w2v_coherence_score = w2v_coherence_model.get_coherence_per_topic()
    w2v_coherence_avg = w2v_coherence_model.get_coherence()
except:
    #In case of OOV Error
    w2v_coherence_score = cv_coherence_score
    w2v_coherence_avg = cv_coherence_avg

# Computing the C_UCI score
c_uci_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_uci')
c_uci_coherence_score = c_uci_coherence_model.get_coherence_per_topic()
c_uci_coherence_avg = c_uci_coherence_model.get_coherence()

# Computing the C_NPMI score
c_npmi_coherence_model = CoherenceModel(topics=topics_text,texts=txtDocs,dictionary=vocab,coherence='c_npmi')
c_npmi_coherence_score = c_npmi_coherence_model.get_coherence_per_topic()
c_npmi_coherence_avg = c_npmi_coherence_model.get_coherence()


logger.write('Topics:\n')

for tp,cv,w2v in zip(topics_text,cv_coherence_score,w2v_coherence_score):
    print('{}+++$+++cv:{}+++$+++w2v:{}'.format(tp,cv,w2v))
    logger.write('{}+++$+++cv:{}+++$+++w2v:{}\n'.format(str(tp),cv,w2v))

print('c_v for ${}$: {}'.format(run_name,cv_coherence_avg))
logger.write('c_v for ${}$: {}\n'.format(run_name, cv_coherence_avg))

print('c_w2v for ${}$: {}'.format(run_name,w2v_coherence_avg))
logger.write('c_w2v for ${}$: {}\n'.format(run_name, w2v_coherence_avg))

print('c_uci for ${}$: {}'.format(run_name,c_uci_coherence_avg))
logger.write('c_uci for ${}$: {}\n'.format(run_name, c_uci_coherence_avg))

print('c_npmi for ${}$: {}'.format(run_name,c_npmi_coherence_avg))
logger.write('c_npmi for ${}$: {}\n'.format(run_name, c_npmi_coherence_avg))

print('t_div for ${}$: {}'.format(run_name,topic_diversity))
logger.write('t_div for ${}$: {}\n'.format(run_name, topic_diversity))


logger.close()

