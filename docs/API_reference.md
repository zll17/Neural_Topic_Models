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
3. TODO: 处理get_topic_word_dist

2023-8-15
1. 增加inference_dataset,几乎等于get_embed
2. utils中增加辅助函数compress_bow和sort_topics
3. 优化get_document_topics
4. getitem添加了接受切片作为参数。主要用于直接取数据来观察，在模型训练中是用不到的。collate_fn会拼合，所有还是不要改int作为输入时的返回值格式（即，没有在外面再嵌套一个维度）。并不能去掉get_embed的num参数，因为不可以把corpus的切片作为输入，他不是Dataset类，只是数据的元组。
   
notes:
1. 注意bow的格式，以前的和gensim的。以前的bow是tensor\[vocab_size\], 或者numpy array，gensim的格式是list, [(token id, freq), (token id, freq), ...]，一定要注意，DocDataset的getitem对此做了转换。不要从zhdd加载以前的数据，要算一组新的，放在新文件夹zhdd_dev中。目前convert_to_BOW都是gensim格式。
2. 想通了内在逻辑：模型相关部分全部是tensor和numpy作为输入输出，外围函数（base_model中的一些）套皮接受list参数
3. 对于格式转换，主要是DocDataset的getitem处理了，即，涉及到需要model的函数的，都要变成DocDataset类，而尽量不要直接输入列表格式的数据。添加的compress_bow目前只在get_document_topics用到，因为一般不会只处理一个数据。

2023-8-17
1. 优化get_topic_word_dist及外围一系列和打印topic有关的函数。减少了函数数量，极致简洁。包括给utils.py中的sort_topics添加topk参数，代替原来的_get_topics()。

### TODO
1. train的参数，去掉test_data
2. 合并inference和inference_by_bow -> 重载 -> done
3. 测试unittest: 用原是代码跑一个版本，作为对比
4. 各个阶段的输出统一
5. example写一个jupyter notebook展示结果
6. 打包至pip
7. 文档用readthedocs
8. ETM_run中的topic_vec和word_vec的功能写成base_model的方法 或者函数
9. train内部分离出一些函数
10. DocDataset.__getitem__()添加切片，这样可以去掉get_embed()参数中的num -> 不行
11. model.load之前，初始化model的时候必须有bow_dim和n_topic参数，能否改为model=ETM(ckpt_path="xxx")?不太好改。

## API Documentation
### DocDataset
#### methods
\_\_getitem\_\_
    会把[(token id, freq), (token id, freq), ...]列表格式的self.bows，转换为每个bow为\[\[token_id1,token_id2,...\],\[freq1,freq2,...\]\]的tensor。
load
    如果是load的corpus，bow中的freq是float型，如果是直接doc2bow得到，是int型。已验证，经过gensim对corpra的gensim.corpora.MmCorpus.serialize保存操作，变成float型了。
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
