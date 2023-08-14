# NTM API Reference
## Usage Examples
简明教程：example.py, example.ipynb
测试：

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

2023-8-13
1. show_topics, show_topic, print_topics, print_topic
    与gensim输出格式一致。不同之处在于添加show_val参数，可选择是否输出word的probability。
    show_topic, print_topics, print_topic放到base model
    原来的show_topic_words()用show_topics()代替。
    show_topics(show_val=False)为默认值，方便类内调用，替代以前的show_topic_words()。其他默认show_val=True。
    TODO: 函数注释放到文档，不写在代码里
2. inference, get_document_topics
   get_document_topics和原本的inference作用一样，get_document_topics调用inference，为了兼顾gensim和原版本的使用习惯才保留了两个。为了兼顾gensim接口。inference的返回值从numpy array改为list。
   inference的返回值为list of float, get_document_topics返回值为按概率降序排列的list of (topic id, probability)。
   为了和gensim接口保持一致，添加__get_item__()，它调用get_document_topics()。
3. TODO: 增加一个inference_batch方法，参考get_embed
4. TODO: 处理get_topic_word_dist

### TODO
1. train的参数，去掉test_data
2. 合并inference和inference_by_bow -> 重载
3. 测试unittest: 用原是代码跑一个版本，作为对比
4. 各个阶段的输出统一
5. example写一个jupyter notebook展示结果
6. 打包至pip
7. 文档用readthedocs

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

print_topic
    return: (topic_id, [(word, value), … ])
    return type: (int, list of (str, float))
    return format same as print_topics, different from gensim.
    gensim:
    return: String representation of topic, like ‘-0.340 * “category” + 0.298 * “$M$” + 0.183 * “algebra” + … ‘.
    return type: str

save
    save tmp files to the given folder

load

### data_utils.py
convert_to_BOW
    相当于dictionary.doc2bow封装。和gensim的dictionary.doc2bow的区别：空句子，或者经过查找字典后为空，需要指出，以便后续inference。dictionary.doc2bow会直接删除空语料。
    TODO: dictionary.doc2bow返回的bow包括原文，跟convert_to_BOW返回的docs重复。考虑去掉docs返回值。
