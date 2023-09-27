#!/usr/bin/env python
# coding: utf-8

import os
import gensim
import pickle
from tokenization import *
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import sys


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


def build_dictionary(docs, no_below=5, no_above=0.1):
    print("Building dictionary ...")
    dictionary = Dictionary(docs)
    # dictionary.filter_n_most_frequent(remove_n=20)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)  # use Dictionary to remove un-relevant tokens
    dictionary.compactify()
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()} # because id2token is empty by default, it is a bug.
    return dictionary


def convert_to_BOW(docs, dictionary, keep_empty_doc=False):
    '''
        param: keep_empty_doc: set True for inference, set False for training
    '''
    print("Converting to BOW ...")
    bows, _docs = [],[]
    for doc in docs:
        if (doc is None or doc==[]) and keep_empty_doc is True:
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
    print(f'Processed {len(bows)} documents.')
    return bows, docs


def compute_tfidf(bows):
    tfidf_model = TfidfModel(bows)
    tfidf = [tfidf_model[bow] for bow in bows]
    return tfidf, tfidf_model


def load_dictionary(txt_path):
    try:
        dictionary = Dictionary.load_from_text(txt_path)
    except:
        print("ERROR: Dictionary path not found. Check again or save corpus before trainning from checkpoint.")
        return None
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()} # because id2token is empty be default, it is a bug.
    return dictionary