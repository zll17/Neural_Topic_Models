<p align="center" id="title_en"><img src="assets/logo.png" width="480"\></p>

[English](#title_en) | [中文](#title_zh)



PyTorch implementations of Neural Topic Model varieties proposed in recent years, including NVDM-GSM, WLDA, WTM-GMM, ETM, BATM ,and GMNTM. The aim of this project is to provide a practical and working example for neural topic models to facilitate the research of related fields. Configuration of the models will not exactly the same as those proposed in the papers, and the hyper-parameters are not carefully finetuned, but I have chosen to get the core ideas covered. 

Empirically, NTM is superior to classical statistical topic models ,especially on short texts. Datasets of short news ([cnews10k](#cnews10k_exp)), dialogue utterances ([zhddline](#zhddline_exp)) and conversation ([zhdd](#zhdd_exp)), are presented for evaluation purpose, all of which are in Chinese. As a comparison to the NTM, an out-of-box LDA script is also provided, which is based on the gensim library. 

Any suggestions or contributions to improving this implementation of NTM are welcomed.

<h2 id="TOC_EN">Table of Contents</h2>

  * [Installation](#Installation)
  * [Models](#Models)
    + [NVDM-GSM](#NVDM-GSM)
    + [WTM-MMD](#WTM-MMD)
    + [WTM-GMM](#WTM-GMM)
    + [ETM](#ETM)
    + [GMNTM](#GMNTM-VaDE)
    + [BATM](#BATM)
* [Datasets](#Datasets)
  * [cnews10k](#cnews10k_exp)
  * [zhddline](#zhddline_exp)
  * [zhdd](#zhdd_exp)
* [Usage](#Usage)
* [Acknowledgement](#Acknowledgement)



<h2 id="Installation">Table of Contents</h2>

```shell
$ git clone https://github.com/zll17/Neural_Topic_Models
$ cd Neural_Topic_Models/
$ sudo pip install -r requirements.txt
```



<h2 id="Models">Models</h2>

<h3 id="NVDM-GSM">NVDM-GSM</h3>

_Discovering Discrete Latent Topics with Neural Variational Inference_

#### Authors
Yishu Miao

#### Description
VAE + Gaussian Softmax

<p align="center">
    <img src="assets/vae_arch.png" width="720"\>
</p>

[[Paper]](http://proceedings.mlr.press/v70/miao17a.html) [[Code]](models/GSM.py)

#### Run Example
```
$ python3 GSM_run.py --taskname cnews10k --n_topic 20 --num_epochs 600 --no_above 0.0134 --criterion cross_entropy --use_fc1
```

<p align="center">
    <img src="assets/GSM_cnews10k.png" width="auto"\>
</p>


<h3 id="WTM-MMD">WTM-MMD</h3>

_Topic Modeling with Wasserstein Autoencoders_

#### Authors
Feng Nan, Ran Ding, Ramesh Nallapati, Bing Xiang

#### Description
WAE with Dirichlet prior + Gaussian Softmax.

[[Paper]](https://www.aclweb.org/anthology/P19-1640/) [[Code]](models/WLDA.py)

#### Run Example
```shell
$ python3 WTM_run.py --taskname cnews10k --n_topic 20 --num_epochs 600 --no_above 0.013 --dist dirichlet
```



<h3 id="WTM-GMM">WTM-GMM</h3>

_Research on Clustering for Subtitle Dialogue Text Based on Neural Topic Model_

#### Authors

Leilan Zhang

#### Abstract

WAE with Gaussian Mixture prior + Gaussian Softmax.

[[Paper]](Under review) [[Code]](models/WLDA.py)



#### Run Example
```shell
$ python3 WTM_run.py --taskname zhdd --n_topic 20 --num_epochs 300 --no_above 0.039 --dist gmm-ctm
```

<p align="center">
    <img src="assets/WLDA-GMM_zhdd.png" width="auto"\>
</p>


<h3 id="ETM">ETM</h3>

_Topic Modeling in Embedding Spaces_

#### Authors
Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei

#### Abstract
VAE + Gaussian Softmax + Embedding

<p align="center">
    <img src="assets/etm_arch.png" width="720"\>
</p>
[[Paper]](https://arxiv.org/abs/1907.04907) [[Code]](models/ETM.py)

#### Run Example
```
$ python3 ETM_run.py --taskname zhdd --n_topic 20 --num_epochs 900 --no_above 0.039
```



<h3 id="GMNTM-VaDE">GMNTM</h3>

_Research on Clustering for Subtitle Dialogue Text Based on Neural Topic Model_

#### Authors
Leilan Zhang

#### Abstract
Based on VaDE ([Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering](https://arxiv.org/abs/1611.05148))

<p align="center">
    <img src="assets/gmvae_arch.png" width="720"\>
</p>


[[Paper]](https://arxiv.org/abs/1611.06430) [[Code]](implementations/ccgan/ccgan.py)

#### Run Example
```
$ python3 GMNTM_run.py --taskname zhdd --n_topic 20 --num_epochs 300 --no_above 0.039 
```



<h3 id="BATM">BATM</h3>

_Neural Topic Modeling with Bidirectional Adversarial Training_

#### Authors
Rui Wang, Xuemeng Hu, Deyu Zhou, Yulan He, Yuxuan Xiong, Chenchen Ye, Haiyang Xu

#### Description

GAN+Encoder

<p align="center">
    <img src="assets/BATM_arch.png" width="720"\>
</p>


[[Paper]](https://arxiv.org/abs/2004.12331) [[Code]](models/BATM.py)

#### Run Example
```
$ python3 BATM_run.py --taskname zhdd --n_topic 20 --num_epochs 300 --no_above 0.039 
```



<h2 id="Datasets">Datasets</h2>

- cnews10k: short cnews sampled from the [cnews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) dataset, in Chinese.
- zhddline: a dialogue dataset in Chinese, translated from the [DailyDialog](https://www.aclweb.org/anthology/I17-1099/) dataset by Sogou translation API.
- zhdd:  Every conversation is concatenated  as a document to be processed. There're 12336 documents in total.
- 3body1: The famous science fiction *The Three-Body Problem*, each paragraph is taken as a document.

​	Basic statistics are listed in the following table:

| dataset  | num of document | genre            | avg len of docs | language |
| -------- | --------------- | ---------------- | --------------- | -------- |
| cnews10k | 10k             | short news       | 18.7            | Chinese  |
| zhddline | 96785           | short utterances | 18.1            | Chinese  |
| zhdd     | 12336           | short dialogues  | 142.1           | Chinese  |
| 3body1   | 2626            | long novel       | 73.8            | Chinese  |

##### Some snippets

<h6 id="cnews10k_exp">cnews10k</h6>

<p align="center">
    <img src="assets/cnews10k_exp.png" width="640"\>
</p>

<h6 id="zhddline_exp">zhddline</h6>

<p align="center">
    <img src="assets/zhddline_exp.png" width="640"\>
</p>

<h6 id="zhdd_exp">zhdd</h6>

<p align="center">
    <img src="assets/zhdd_exp.png" width="640"\>
</p>

<h6 id="3body1_exp">3body1</h6>

<p align="center">
    <img src="assets/3body1_exp.png" width="720"\>
</p>





<h4 id="Usage">Usage</h4>



<h4 id="Acknowledgement">Acknowledgement</h4>

A big part of this project is supported by my supervisor Prof. Qiang Zhou, I would highly appreciate him for his valuable comments and helpful suggestions.

In the construction of this project, some implementations are taken as reference, I would like to thank the contributors of those projects: VaDE, WLDA, ETM.

<h4 id="License">License</h4>

Apache License 2.0



-------------------

<p align="center" id="title_zh"><img src="assets/logo.png" width="480"\></p>

[English](#title_en) | [中文](#title_zh)



一些神经主题模型（Neural Topic Model, NTM）的PyTorch实现，包括NVDM-GSM、WLDA、WTM-GMM、ETM、 BATM 和 GMNTM。

近年来基于VAE和GAN的神经主题模型的各类变种，相比于经典的统计主题模型（如LDA等），能提取到更一致的主题。NTM在稀疏性十分严重的场景下所提取到的主题，其一致性和多样性都优于LDA，是一种强大（且有意思）的模型。此项目的初衷是提供一组方便、实用的神经主题模型实现，包括部分论文的复现及我自己的改进模型。项目中的模型配置与原论文未必完全一致，但保留了原论文中心思想。

此项目提供有三个中文短文本数据集——新闻标题数据集（[cnews10k](cnews10k)）和对话数据集（[zhdd](zhdd) 和 [zhddline](zhddline)），作评测之用。作为对比，提供了基于gensim编写的LDA脚本，开箱即用，且接口与NTM保持一致。

## 目录

  * [TODO](TODO)
  * [安装](#安装)
  * [模型](模型)
  * [数据集](数据集)
  * [应用示例](应用示例)
  * [致谢](致谢)



## TODO

- 训练模型权重保存
- log 曲线绘制
- 文档-主题分布推断
- ETM 主题向量、词向量获取、保存
- 隐空间绘制
- 中文文档完善



## 安装

``` shell
$ git clone https://github.com/zll17/Neural_Topic_Models
$ cd Neural_Topic_Models/
$ sudo pip install -r requirements.txt
```



## 模型

### NVDM-GSM

论文： _Discovering Discrete Latent Topics with Neural Variational Inference_

#### Authors

Yishu Miao

#### Description

VAE + Gaussian Softmax

<p align="center">
    <img src="assets/vae_arch.png" width="720"\>
</p>

