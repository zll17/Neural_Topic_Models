# NTM API Reference
## Usage Examples
See example.py

## API
Class:
DocDatasets
model

## New Version Features
### 修改原则
1. 新旧版本兼容 [to discuss]
2. 最大化解耦各个部分
3. model部分提炼出一个base model
4. 对标gensim的api

### 修改历史
2023-8-11
1. inference
   以ETM为例。原来在inference中处理token到bow的过程，逐个token使用dictionary.token2id，就是为了防止oov。
   现在在外层使用convert_to_BOW，其经过升级已经能够处理各种情况，inference接受bow作为参数，与gensim一致。
   去掉了dictionary参数。


### TODO
1. train的参数，去掉test_data

## API Documentation
### DocDataset
#### methods
load

Helper functions in `data_utils.py`
Update: Removed TestDocDataset

### model
#### attributes

#### methods
train
    parameters:
        log_every: saves ckpt *and evaluates* every log_every steps
    TODO: train的参数去掉test_data

evaluate
    evaluate on traning corpus. question: can it evaluate on other unseen corpus?
    return: criteria (values)

inference
    return: probabilities

save
    save tmp files to the given folder

load

### data_utils.py
convert_to_BOW
    相当于dictionary.doc2bow封装。和gensim的dictionary.doc2bow的区别：空句子，或者经过查找字典后为空，需要指出，以便后续inference。dictionary.doc2bow会直接删除空语料。
    TODO: dictionary.doc2bow返回的bow包括原文，跟convert_to_BOW返回的docs重复。考虑去掉docs返回值。
