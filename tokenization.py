import re

from pyhanlp import *

def Tokenizer(sent, stopwords=None):
    pat = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@—，。：★、￥…【】（）《》？“”‘’！\[\\\]^_`{|}~\u3000]+')
    tokens = [t.word for t in HanLP.segment(sent)]
    tokens = [re.sub(pat, r'', t).strip() for t in tokens]
    tokens = [t for t in tokens if t != '']
    if stopwords is not None:
        tokens = [t for t in tokens if not (t in stopwords)]
    return tokens


'''
def Tokenizer(sent,stopwords=None):
    # Tokenizer for English. 
    pat = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@—，。：★、￥…【】（）《》？“”‘’！\[\\\]^_`{|}~\u3000]+')
    tokens = [re.sub(pat,r'',t).strip() for t in sent.split(' ')]
    tokens = [t for t in tokens if t!='']
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(t).lower() for t in tokens]    
    if stopwords is not None:
        tokens = [t for t in tokens if not (t in stopwords)]                                               
    return tokens
'''

if __name__ == '__main__':
    print(Tokenizer('他拿的是《红楼梦》？！我还以为他是个Foreigner———'))