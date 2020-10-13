<p align="center"><img src="assets/logo.png" width="480"\></p>

PyTorch implementations of Neural Topic Model varieties proposed in recent years, including NVDM-GSM, WLDA, ETM, BATM ,and GMNTM. The aim of this project is to provide a practical and working example for neural topic models to facilitate the research of related fields. Configuration of the models will not exactly the same as those proposed in the papers, and the hyper-parameters are not carefully finetuned, but I have chosen to get the core ideas covered. 

Empirically, NTM is superior to classical statistical topic models ,especially on short texts. Datasets of short news ([cnews10k](#cnews10k)) and dialogue utterances ([zhdd](#zhdd)), both of which are in Chinese, are presented for evaluation purpose. As a comparison to the NTM, an out-of-box LDA script is also provided, which is based on the gensim library. 

Any suggestions or contributions to improving this implementation of NTM are welcomed.

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [NVDM-GSM](#NVDM-GSM)
    + [WLDA-MMD](#WLDA-MMD)
    + [WLDA-GMM](WLDA-GMM)
    + [ETM](#ETM)
    + [GMNTM](#GMNTM-VaDE)
    + [BATM](#BATM)
* [Datasets](#Datasets)
  * [cnews10k](#cnews10k)
  * [zhdd](#zhdd)
  * [zhddline](zhddline)
* [Usage](#usage)
* [Demos](#Demos)
* [Acknowledgement](#Acknowledgement)

## Installation
```shell
$ git clone https://github.com/zll17/Neural_Topic_Models
$ cd Neural_Topic_Models/
$ sudo pip install -r requirements.txt
```

## Implementations   
### NVDM-GSM
_Discovering Discrete Latent Topics with Neural Variational Inference_

#### Authors
Yishu Miao

#### Description
VAE + Gaussian Softmax

<p align="center">
    <img src="assets/vae_arch.png" width="auto"\>
</p>

[[Paper]](http://proceedings.mlr.press/v70/miao17a.html) [[Code]](models/GSM.py)

#### Run Example
```
$ python3 GSM_run.py --taskname cnews10k --n_topic 20 --num_epochs 600 --no_above 0.0134 --criterion cross_entropy --use_fc1
```

<p align="center">
    <img src="assets/GSM_cnews10k.png" width="auto"\>
</p>



### WLDA-MMD
_Topic Modeling with Wasserstein Autoencoders_

#### Authors
Feng Nan, Ran Ding, Ramesh Nallapati, Bing Xiang

#### Description
WAE with Dirichlet prior + Gaussian Softmax.

[[Paper]](https://www.aclweb.org/anthology/P19-1640/) [[Code]](models/WLDA.py)

#### Run Example
```
$ python3 WLDA_run.py --taskname cnews10k --n_topic 20 --num_epochs 600 --no_above 0.013 --dist dirichlet
```



### WLDA-GMM
_Research on Clustering for Subtitle Dialogue Text Based on Neural Topic Model_

#### Authors

Leilan Zhang

#### Abstract

WAE with Gaussian Mixture prior + Gaussian Softmax.

[[Paper]](Under review) [[Code]](models/WLDA.py)



#### Run Example
```shell
$ python3 WLDA_run.py --taskname zhdd --n_topic 20 --num_epochs 300 --no_above 0.039 --dist gmm-ctm
```

<p align="center">
    <img src="assets/WLDA-GMM_zhdd.png" width="auto"\>
</p>



### ETM
_Topic Modeling in Embedding Spaces_

#### Authors
Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei

#### Abstract
VAE + Gaussian Softmax + Embedding

<p align="center">
    <img src="assets/etm_arch.png" width="auto"\>
</p>

[[Paper]](https://arxiv.org/abs/1907.04907) [[Code]](models/ETM.py)

#### Run Example
```
$ python3 ETM_run.py --taskname zhdd --n_topic 20 --num_epochs 900 --no_above 0.039
```



### GMNTM

_Research on Clustering for Subtitle Dialogue Text Based on Neural Topic Model_

#### Authors
Leilan Zhang

#### Abstract
Based on VaDE ([Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering](https://arxiv.org/abs/1611.05148))

<p align="center">
    <img src="assets/gmvae_arch.png" width="auto"\>
</p>

[[Paper]](https://arxiv.org/abs/1611.06430) [[Code]](implementations/ccgan/ccgan.py)

#### Run Example
```
$ python3 GMNTM_run.py --taskname zhdd --n_topic 20 --num_epochs 300 --no_above 0.039 
```



### BATM

_Neural Topic Modeling with Bidirectional Adversarial Training_

#### Authors
Rui Wang, Xuemeng Hu, Deyu Zhou, Yulan He, Yuxuan Xiong, Chenchen Ye, Haiyang Xu

#### Abstract
Recent years have witnessed a surge of interests of using neural topic models for automatic topic extraction from text, since they avoid the complicated mathematical derivations for model inference as in traditional topic models such as Latent Dirichlet Allocation (LDA). However, these models either typically assume improper prior (e.g. Gaussian or Logistic Normal) over latent topic space or could not infer topic distribution for a given document. To address these limitations, we propose a neural topic modeling approach, called Bidirectional Adversarial Topic (BAT) model, which represents the first attempt of applying bidirectional adversarial training for neural topic modeling. The proposed BAT builds a two-way projection between the document-topic distribution and the document-word distribution. It uses a generator to capture the semantic patterns from texts and an encoder for topic inference. Furthermore, to incorporate word relatedness information, the Bidirectional Adversarial Topic model with Gaussian (Gaussian-BAT) is extended from BAT. To verify the effectiveness of BAT and Gaussian-BAT, three benchmark corpora are used in our experiments. The experimental results show that BAT and Gaussian-BAT obtain more coherent topics, outperforming several competitive baselines. Moreover, when performing text clustering based on the extracted topics, our models outperform all the baselines, with more significant improvements achieved by Gaussian-BAT where an increase of near 6\% is observed in accuracy.

#### Description

<p align="center">
    <img src="assets/BATM_arch.png" width="auto"\>
</p>

[[Paper]](https://arxiv.org/abs/2004.12331) [[Code]](models/BATM.py)

#### Run Example
```
$ python3 BATM_run.py --taskname zhdd --n_topic 20 --num_epochs 300 --no_above 0.039 
```



### Datasets

- cnews10k: short cnews sampled from the [cnews]() dataset, in Chinese.
- zhddline: a dialogue dataset in Chinese, translated from the [DailyDialog](https://www.aclweb.org/anthology/I17-1099/) dataset by Sogou translation API.
- zhdd:  Every dialogue is concatenated to be processed as a document. There're 12336 documents in total.
- 3body1: The famous science fiction *The Three-Body Problem*, each paragraph is taken as a document.

​	Basic statistics are listed in the following table:

| dataset  | num of document | genre            | avg len of docs | language |
| -------- | --------------- | ---------------- | --------------- | -------- |
| cnews10k | 10k             | short news       | TD              | Chinese  |
| zhddline | TD              | short utterances | TD              | Chinese  |
| zhdd     | TD              | short dialogues  | TD              | Chinese  |
| 3body1   | TD              | long novel       | TD              | Chinese  |

#### Usage

TODO

#### Demos

TODO

#### Acknowledgement

A big part of this project is taken from my master thesis, which is supported by my supervisor Prof. Qiang Zhou, I would like to thank him for valuable comments and helpful suggestions 

#### License

TBD



