import pickle
import re
import argparse
from data import Vocabulary
from utils import *

parser = argparse.ArgumentParser(description='Calculate the Topic Coherence and Topic Diversity value')
parser.add_argument('--taskname',type=str,default='sub',help='taskname')

args = parser.parse_args()

bows = pickle.load(open('data/{}/bows.pkl'.format(args.taskname),'rb'))
vocab = pickle.load(open('data/{}/vocab.pkl'.format(args.taskname),'rb'))


tokens = bows['tokens']

topn = []
with open('{}_result.log'.format(args.taskname),'r',encoding='utf-8') as rfp:
    pat = re.compile(r'.*?\[(.*?)\]')
    for line in rfp.readlines():
        mat = re.match(pat,line)
        if mat:
            topn.append([t.strip().strip("'") for t in str(mat.group(1)).split(',')])
print(topn)
print('Computing Topic Coherence...')
print(calc_topic_coherence(topn,tokens,vocab))
print('Computing Topic Diversity...')
print(calc_topic_diversity(topn))
