#!/usr/bin/env python
# coding: utf-8

import os
import gensim
import pickle
from tokenization import *
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import sys


def load_tmp_files(load_dir, use_tfidf=False):
    print("Loading corpus from %s ..."%load_dir)
    bows = gensim.corpora.MmCorpus(os.path.join(load_dir,'corpus.mm'))
    tfidf = gensim.corpora.MmCorpus(os.path.join(load_dir,'tfidf.mm')) if use_tfidf else None
    dictionary = Dictionary.load_from_text(os.path.join(load_dir,'dict.txt'))
    docs = pickle.load(open(os.path.join(load_dir,'docs.pkl'),'rb'))
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()} # because id2token is empty be default, it is a bug.
    return dictionary, bows, docs, tfidf


def file_tokenize(txt_path, lang, stopwords=None, tokenizer=None):
    '''
        txtLines is the list of string, without any preprocessing.
        texts is the list of list of tokens.
    '''
    print('Tokenizing ...')
    txtLines = [line.strip('\n') for line in open(txt_path,'r',encoding='utf-8')]
    if stopwords==None:
        stopwords = set([l.strip('\n').strip() for l in open(os.path.join(os.getcwd(),'data','stopwords.txt'),'r',encoding='utf-8')])
    if tokenizer is None:
        tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)
    docs = tokenizer.tokenize(txtLines)
    docs = [line for line in docs if line!=[]]
    return docs


def build_dictionary(docs, save_dir=None, no_below=5, no_above=0.1):
    print("Building dictionary ...")
    dictionary = Dictionary(docs)
    # dictionary.filter_n_most_frequent(remove_n=20)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)  # use Dictionary to remove un-relevant tokens
    dictionary.compactify()
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()} # because id2token is empty by default, it is a bug.
    if save_dir:
        dictionary.save_as_text(os.path.join(save_dir,'dict.txt'))
        print("Dictionary saved to %s"%os.path.join(save_dir,'dict.txt'))
    return dictionary


def convert_to_BOW(docs, dictionary, save_dir=None, keep_empty_doc=False):
    '''
        param: keep_empty_doc: set True for inference, set False for training
    '''
    print("Converting to BOW ...")
    bows, _docs = [],[]
    for doc in docs:
        if doc is None and keep_empty_doc is True:
            _docs.append(None)
            bows.append(None)     
        else:           
            _bow = dictionary.doc2bow(doc)
            if _bow!=[]:
                _docs.append(list(doc))
                bows.append(_bow)
            elif keep_empty_doc:
                _docs.append(None)
                bows.append(None)                      
    docs = _docs
    if save_dir:
        gensim.corpora.MmCorpus.serialize(os.path.join(save_dir,'corpus.mm'), bows)
        print("Corpus saved to %s"%os.path.join(save_dir,'corpus.mm'))
        pickle.dump(docs,open(os.path.join(save_dir,'docs.pkl'),'wb'))
        print("Docs saved to %s"%os.path.join(save_dir,'docs.pkl'))
    print(f'Processed {len(bows)} documents.')
    return bows, docs


def compute_tfidf(bows, save_dir=None):
    tfidf_model = TfidfModel(bows)
    tfidf = [tfidf_model[bow] for bow in bows]
    if save_dir:
        gensim.corpora.MmCorpus.serialize(os.path.join(save_dir,'tfidf.mm'),tfidf)
        print("TF-IDF model saved to %s"%os.path.join(save_dir,'tfidf.mm'))
    return tfidf, tfidf_model


def load_dictionary(txt_path):
    return Dictionary.load_from_text(txt_path)