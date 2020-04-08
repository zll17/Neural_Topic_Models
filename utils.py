import torch 
import numpy as np

def calc_topic_diversity(topn_words):
    '''topn_words is [[w11,w12,...,w1n],[w21,w22,...,w2n],...]'''
    voc = set([])
    for l in topn_words:
        for w in l:
            voc.add(w)
    n_total = len(topn_words)*len(topn_words[0])
    TD = len(voc)/n_total
    print('Topic diversity is: {}'.format(TD))
    return TD

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))
    return TD

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            #doc = data[l].squeeze(0)
            doc = data[l]
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        #doc = data[l].squeeze(0)
        doc = data[l]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def calc_topic_coherence(topn_words,data,vocab):
    ids = [[vocab.token2id[w] for w in l] for l in topn_words]
    D = len(data)
    print('D: ', D)
    TC = []
    num_topics = len(topn_words)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = ids[k]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = abs(np.mean(TC) / counter)
    print('Topic coherence is: {}'.format(TC))
    return TC



def get_topic_coherence(beta, data, vocab):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab.id2word[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = abs(np.mean(TC) / counter)
    print('Topic coherence is: {}'.format(TC))
    return TC

def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.token2id[word]
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab.id2word[comp] for comp in nearest_neighbors]
    return nearest_neighbors
