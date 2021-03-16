import re
from typing import List
from collections import defaultdict
import multiprocessing

from tqdm import tqdm

from pyhanlp import *
import spacy

LANG_CLS = defaultdict(lambda:"SpacyTokenizer")
LANG_CLS.update({
    "zh": "HanLPTokenizer",
    "en": "SpacyTokenizer",
})

SPACY_MODEL = {
    "en": "en_core_web_sm",
    "ja": "ja_core_news_sm"
}


class HanLPTokenizer(object):
    def __init__(self, stopwords=None):
        self.pat = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@—，。：★、￥…【】（）《》？“”‘’！\[\\\]^_`{|}~\u3000]+')
        self.stopwords = stopwords
        print("Using HanLP tokenizer")
        
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        docs = []
        for line in tqdm(lines):
            tokens = [t.word for t in HanLP.segment(line)]
            tokens = [re.sub(self.pat, r'', t).strip() for t in tokens]
            tokens = [t for t in tokens if t != '']
            if self.stopwords is not None:
                tokens = [t for t in tokens if not (t in self.stopwords)]
            docs.append(tokens)
        return docs
        
        
class SpacyTokenizer(object):
    def __init__(self, lang="en", stopwords=None):
        self.stopwords = stopwords
        self.nlp = spacy.load(SPACY_MODEL[lang], disable=['ner', 'parser'])
        print("Using SpaCy tokenizer")

        
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        docs = self.nlp.pipe(lines, batch_size=1000, n_process=multiprocessing.cpu_count())
        docs = [[token.lemma_ for token in doc if not (token.is_stop or token.is_punct)] for doc in docs]
        return docs
        

if __name__ == '__main__':
    tokenizer=HanLPTokenizer()
    print(tokenizer.tokenize(['他拿的是《红楼梦》？！我还以为他是个Foreigner———']))
