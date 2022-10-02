#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import torch

class BaseNTM(object):
    def __init__(self, name: str):
        self.name = name
        self.dict_path = None
        self.param = {}
        self.state = {}
        self.save_dir = None

    # def train(self):
    #     pass
    
    # def update(self):
    #     ''' 
    #     Update (train) model incrementally with unseen documents.
    #     Adapted from gensim.
    #     Add or not?
    #     '''
    #     pass

    # def get_embed(self):
    #     pass
    
    # def get_topic_word_dist(self):
    #     ''' 
    #     gensim: __getitem__
    #     '''
    #     pass
    
    # def show_topic_words(self):
    #     '''
    #     gensim: __getitem__
    #     '''
    #     pass
 
    # def evaluate(self):
    #     pass
    
    # def diff(self):
    #     '''
    #     gensim: Calculate the difference in topic distributions between two models: self and other.
    #     Add or not?
    #     '''
    #     pass

    # def inference(self):
    #     pass
    
    # def inference_by_bow(self):
    #     pass
    
    def _update_param(self, **kwargs):
        self.param.update(kwargs)

    def _update_state(self, **kwargs):
        self.state.update(kwargs)

    def save(self, save_path: str = None):
        '''
            If called by user, need to specify the save_path.
            If called by subclass, save_path is set by default.
        '''
        if save_path is None:
            if self.save_dir is None:
                folder_name = f'{time.strftime("%m-%d-%H-%M", time.localtime())}_{self.name}_tp{self.param["n_topic"]}'
                self.save_dir = os.path.join(os.getcwd(),'ckpt',folder_name)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            save_path = os.path.join(self.save_dir, "ep%d.ckpt"%self.state["epoch"])

        ckpt = {"state": self.state, "param": self.param}
        torch.save(ckpt, save_path)
        print("Checkpoint saved to %s"%save_path)