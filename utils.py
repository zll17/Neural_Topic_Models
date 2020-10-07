#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/10/05 13:46:04
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''


import os
import gensim
import numpy as np
from gensim.models.coherencemodel import CoherenceModel

def get_topic_words(model,topn=15,n_topic=10,vocab=None,fix_topic=None,showWght=False):
    topics = []
    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]],t[1]) for t in model.get_topic_terms(tp_idx,topn=topn)]
        else:
            return [vocab.id2token[t[0]] for t in model.get_topic_terms(tp_idx,topn=topn)]
    if fix_topic is None:
        for i in range(n_topic):
            topics.append(show_one_tp(i))
    else:
        topics.append(show_one_tp(fix_topic))
    return topics

def calc_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div

def calc_topic_coherence(topic_words,docs,dictionary,emb_path=None,taskname=None,sents4emb=None,calc4each=False):
    # emb_path: path of the pretrained word2vec weights, in text format.
    # sents4emb: list/generator of tokenized sentences.
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_v')
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()
    
    # Computing the C_W2V score
    try:
        w2v_model_path = os.path.join(os.getcwd(),'data',f'{taskname}','w2v_weight_kv.txt')
        # Priority order: 1) user's embed file; 2) standard path embed file; 3) train from scratch then store.
        if emb_path!=None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path,binary=False)
        elif os.path.exists(w2v_model_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary=False)
        elif sents4emb!=None:
            print('Training a word2vec model 20 epochs to evaluate topic coherence, this may take a few minutes ...')
            w2v_model = gensim.models.Word2Vec(sents4emb,size=300,min_count=1,workers=6,iter=20)
            keyed_vectors = w2v_model.wv
            keyed_vectors.save_word2vec_format(w2v_model_path,binary=False)
        else:
            raise Exception("C_w2v score isn't available for the missing of training corpus (sents4emb=None).")
            
        w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v',keyed_vectors=keyed_vectors)

        w2v_per_topic = w2v_coherence_model.get_coherence_per_topic() if calc4each else None
        w2v_score = w2v_coherence_model.get_coherence()
    except Exception as e:
        print(e)
        #In case of OOV Error
        w2v_per_topic = [None for _ in range(len(topic_words))]
        w2v_score = None
    
    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_uci')
    c_uci_per_topic = c_uci_coherence_model.get_coherence_per_topic() if calc4each else None
    c_uci_score = c_uci_coherence_model.get_coherence()
    
    
    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_npmi')
    c_npmi_per_topic = c_npmi_coherence_model.get_coherence_per_topic() if calc4each else None
    c_npmi_score = c_npmi_coherence_model.get_coherence()
    return (cv_score,w2v_score,c_uci_score, c_npmi_score),(cv_per_topic,w2v_per_topic,c_uci_per_topic,c_npmi_per_topic)

def mimno_topic_coherence(topic_words,docs):
    tword_set = set([w for wlst in topic_words for w in wlst])
    word2docs = {w:set([]) for w in tword_set}
    for docid,doc in enumerate(docs):
        doc = set(doc)
        for word in tword_set:
            if word in doc:
                word2docs[word].add(docid)
    def co_occur(w1,w2):
        return len(word2docs[w1].intersection(word2docs[w2]))+1
    scores = []
    for wlst in topic_words:
        s = 0
        for i in range(1,len(wlst)):
            for j in range(0,i):
                s += np.log((co_occur(wlst[i],wlst[j])+1.0)/len(word2docs[wlst[j]]))
        scores.append(s)
    return np.mean(s)

def evaluate_topic_quality(topic_words, test_data, taskname=None, calc4each=False):
    
    td_score = calc_topic_diversity(topic_words)
    print(f'topic diversity:{td_score}')
    
    (c_v, c_w2v, c_uci, c_npmi),\
        (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic) = \
        calc_topic_coherence(topic_words=topic_words, docs=test_data.docs, dictionary=test_data.dictionary,
                             emb_path=None, taskname=taskname, sents4emb=test_data, calc4each=calc4each)
    print('c_v:{}, c_w2v:{}, c_uci:{}, c_npmi:{}'.format(
        c_v, c_w2v, c_uci, c_npmi))
    scrs = {'c_v':cv_per_topic,'c_w2v':c_w2v_per_topic,'c_uci':c_uci_per_topic,'c_npmi':c_npmi_per_topic}
    if calc4each:
        for scr_name,scr_per_topic in scrs.items():
            print(f'{scr_name}:')
            for t_idx, (score, twords) in enumerate(zip(scr_per_topic, topic_words)):
                print(f'topic.{t_idx+1:>03d}: {score} {twords}')
    
    mimno_tc = mimno_topic_coherence(topic_words, test_data.docs)
    print('mimno topic coherence:{}'.format(mimno_tc))
    if calc4each:
        return (c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score), (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic)
    else:
        return c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for pt in points:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev*factor+pt*(1-factor))
        else:
            smoothed_points.append(pt)
    return smoothed_points