#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Time    :   2021/01/17
@Author  :   NekoMt.Tai
@Version :   1.0
@Contact :  
@Desc    :   None
'''



import os
import torch
import argparse
import time
from models import BATM, ETM, GMNTM, GSM, WTM
from dataset import TestData
from tqdm import tqdm
import json
from gensim.corpora import Dictionary
from device_helper import default_device


def _load_checkpoint(path, map_location):
    ck = torch.load(path, map_location=map_location)
    if isinstance(ck, dict) and 'param' in ck and 'net' in ck:
        return ck
    raise SystemExit(
        'Checkpoint must be a dict with keys "param" and "net" (as saved by updated *_run.py scripts). '
        'Raw state_dict-only files are not supported here.'
    )


parser = argparse.ArgumentParser('Topic model inference')
parser.add_argument('--no_below',type=int,default=5,help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above',type=float,default=0.005,help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--use_tfidf',action='store_true',help='Use TF-IDF features for BOW input')
parser.add_argument('--model_path',type=str,default='',help='Load model for inference from this path')
parser.add_argument('--save_dir',type=str,default='./',help='Save inference result')
parser.add_argument('--model_name',type=str,default='WTM',help='Neural Topic Model name')
parser.add_argument('--test_path',type=str,default='',help='Test set path')
parser.add_argument(
    '--lang',
    type=str,
    default='zh',
    help='Tokenizer/language for TestData (should match training, e.g. zh for HanLP).',
)

args = parser.parse_args()


def main():
    use_tfidf = args.use_tfidf
    model_path = args.model_path
    model_name = args.model_name
    save_dir = args.save_dir
    test_path = args.test_path
    lang = args.lang

    device = default_device()

    checkpoint = _load_checkpoint(model_path, map_location=device)

    taskname = checkpoint['param']['taskname']
    cwd = os.getcwd()
    tmpDir = os.path.join(cwd,'data',taskname)
    if os.path.exists(os.path.join(tmpDir,'corpus.mm')):
        dictionary = Dictionary.load_from_text(os.path.join(tmpDir,'dict.txt'))
    else:
        raise RuntimeError('Build corpus first (expected {} and dict).'.format(tmpDir))

    testSet = TestData(
        dictionary=dictionary,
        lang=lang,
        txtPath=test_path,
        no_below=args.no_below,
        no_above=args.no_above,
        use_tfidf=use_tfidf,
    )

    param = dict(checkpoint['param'])
    param.update({'device': device})
    Model = globals()[model_name]
    model = Model(**param)
    model.load_model(checkpoint['net'])

    infer_topics = []
    for doc in tqdm(testSet):
        if doc == [] or doc is None:
            infer_topics.append(None)
        else:
            infer_topics.append(model.inference(doc_tokenized=doc, dictionary=dictionary).tolist())

    for i, topic in enumerate(model.show_topic_words(dictionary=dictionary)):
        print("topic{}: {}".format(i, str(topic)))

    with open(test_path, 'r', encoding='utf-8') as f:
        for i in range(10):
            line = f.readline()
            pred = infer_topics[i] if i < len(infer_topics) else None
            if pred is None:
                print(line.strip(), '(no topics: empty or OOV document)')
            else:
                ranked = sorted(enumerate(pred), key=lambda x: x[1], reverse=True)
                print(
                    line.strip(),
                    ' + '.join(['topic{}*{}'.format(j, round(w, 3)) for j, w in ranked]),
                )

    out_path = os.path.join(
        save_dir,
        'inference_result_{}_{}.json'.format(model_name, time.strftime('%Y-%m-%d-%H-%M', time.localtime())),
    )
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(infer_topics, fp)
    print('Inference result saved to', out_path)


if __name__ == "__main__":
    main()
