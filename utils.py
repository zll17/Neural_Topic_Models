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
import torch
import matplotlib.pyplot as plt

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

def train_word2vec(sentences, save_path):
    print('Training a word2vec model 20 epochs to evaluate topic coherence, this may take a few minutes ...')
    w2v_model = gensim.models.Word2Vec(sentences,size=300,min_count=1,workers=6,iter=20)
    keyed_vectors = w2v_model.wv
    keyed_vectors.save_word2vec_format(save_path,binary=False)
    print("Test set word2vec weights saved to %s"%save_path)
    return keyed_vectors

def calc_topic_coherence(topic_words,docs,dictionary,emb_path=None,sents4emb=None,calc4each=False):
    # emb_path: path of the pretrained word2vec weights, in text format.
    # sents4emb: list/generator of tokenized sentences.
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_v')
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()
    
    # Computing the C_W2V score
    try:
        # Priority order: 1) user's specified embed file;  2) train from scratch then store.        
        if emb_path!=None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path,binary=False)
        elif sents4emb!=None:
            keyed_vectors = train_word2vec(sents4emb, emb_path)
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

def evaluate_topic_quality(topic_words, test_data, emb_path=None, calc4each=False):
    
    td_score = calc_topic_diversity(topic_words)
    print(f'topic diversity:{td_score}')
    
    (c_v, c_w2v, c_uci, c_npmi),\
        (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic) = \
        calc_topic_coherence(topic_words=topic_words, docs=test_data.docs, dictionary=test_data.dictionary,
                             emb_path=emb_path, sents4emb=test_data, calc4each=calc4each)
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

# def compress_bow(bow_np) -> List[List[(int, float)]]:
#     pass

def expand_bow(bow_li, bow_dim):
    '''Convert list of bows in gensim format [(token_idx, freq), (token_idx, freq), ...] to 2-d torch tensor
    :param bow_li: List[List[(int, float)]] for many docs, List[(int, float)] for one doc
    :param bow_dim: int
    :return bow: 2-d tensor in shape (num_docs, vocab_size)
    '''
    if bow_li is None or bow_li==[]:
        return np.empty(0)
    if not isinstance(bow_li[0], list):  # if input is bow of one doc
        bow_li = [bow_li]
    n=len(bow_li)
    bow = torch.zeros(n, bow_dim)
    for i in range(n):
        item = list(zip(*bow_li[i])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[i, list(item[0])] = torch.tensor(list(item[1])).float()
    return bow

def sort_topics(topics, topk=None, min_prob=None):
    '''Descending sorting document - topic distribution and topic - word distribution.
    :param topics: numpy array in shape (num_doc, num_topics)
    :return: list of [(topic id, probability),...]
    '''
    if topics.ndim==1:
        topics = np.expand_dims(topics, axis=0)
    if topk is None:
        topk = topics.shape[1]
    topics_sorted = np.sort(topics, axis=1)[:, ::-1][:, :topk].tolist()  # [:, ::-1] to reverse, acsending -> descending
    id_sorted = np.argsort(-topics, axis=1)[:, :topk].tolist()
    res = []
    for i in range(topics.shape[0]):
        if min_prob:
            res.append([(topic_id, prob) for (topic_id, prob) in zip(id_sorted[i], topics_sorted[i]) if prob > min_prob])
        else:
            res.append([(topic_id, prob) for (topic_id, prob) in zip(id_sorted[i], topics_sorted[i])])
    return res

def plot_scores(scrs, log_every, save_file):
    '''
    # From previous version
    for scr_name,scr_lst in scrs.items():
        plt.cla()
        plt.plot(np.array(range(len(scr_lst)))*log_every,scr_lst)
        plt.savefig(f'wlda_{scr_name}.png')
    '''
    plt.cla()
    for scr_name,scr_lst in scrs.items():
        if scr_name in ['c_v','c_w2v','td']:
            plt.plot(np.array(range(len(scr_lst)))*log_every,scr_lst,label=scr_name)
    plt.title('Topic Coherence')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(save_file)

def plot_train_loss(trainloss_lst, log_every, save_file):
    plt.cla()
    smth_pts = smooth_curve(trainloss_lst)
    plt.plot(np.array(range(len(smth_pts)))*log_every,smth_pts)
    plt.xlabel('epochs')
    plt.title('Train Loss')
    plt.savefig(save_file)