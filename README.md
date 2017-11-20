# MatchZoo
----
MatchZoo is a toolkit for text matching. It was developed with a focus on facilitate the designing, comparing and sharing of deep text matching models. 

## Overview
The architecture of the MatchZoo toolit is depicited in Figure 
<div align='center'>
<img src="./data/matchzoo.png" width = "400" height = "200" alt="图片名称" align=center />
</div>
There are three major modules in the toolkit, namely data preparation, model construction, training and evaluation, respectively. These three modules are actually organized as a pipeline of data flow.

### Data Preparation
The data preparation module aims to convert dataset of different text matching tasks into a unified format as the input of deep matching models. Users provide datasets which contains pairs of texts along with their labels, and the module produces the following files.

+	**Word Dictionary**: recordsthemappingfromeachwordto a unique identi er called wid. Words that are too frequent (e.g. stopwords), too rare or noisy (e.g. fax numbers) can be  ltered out by prede ned rules.
+	**Corpus File**: records the mapping from each text to a unique identi er called tid, along with a sequence of word identi ers contained in that text. Note here each text is truncated or padded to a  xed length customized by users.
+	**Relation File**: is used to store the relationship between two texts, each line containing a pair of tids and the cor- responding label.
+ **Input Data Format**: a detailed explaination of input data format can be found in MatchZoo/data/example/readme.md.

### Model Construction
In the model construction module, we employ Keras libarary to help users build the deep matching model layer by layer conveniently.  e Keras libarary provides a set of common layers widely used in neural models, such as convolutional layer, pooling layer, dense layer and so on. To further facilitate the construction of deep text matching models, we extend the Keras libarary to provide some layer interfaces speci cally designed for text matching. 

Moreover, the toolkit has implemented two schools of representative deep text matching models, namely representation-focused models and interactive-focused models[[1]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf).

### Training and Evaluation
For learning the deep matching models, the toolkit provides a variety of objective functions for regression, classification and ranking. For example, the ranking-related objective functions include several well-known pointwise, pairwise and listwise losses. It is flexible for users to pick up di erent objective functions in the training phase for optimization. Once a model has been trained, the toolkit could be used to produce a matching score, predict a matching label, or rank target texts (e.g., a document) against an input text.

## Models

1. DRMM
2. MatchPyramid
3. ARC-I
4. DSSM
5. CDSSM
6. ARC-II
7. MV-LSTM


## Usage
```
git clone https://github.com/faneshion/MatchZoo.git
cd MatchZoo
python setup.py install

python main.py --phase train --model_file ./models/arci_ranking.config
python main.py --phase predict --model_file ./models/arci_ranking.config
```
## Environment
* python2.7+ 
* tensorflow 1.2
* keras 2.05
* tqdm 4.19.4

## Model Detail:

1. DRMM
-------
this model is an implementation of <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.

- model file: models/drmm.py
- config file: models/drmm.config

2. MatchPyramid
-------
this model is an implementation of <a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>

- model file: models/matchpyramid.py
- config file: models/matchpyramid.config

3. ARC-I
-------
this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: models/arci.py
- model config: models/arci.config

4. DSSM
-------
this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

- model file: models/dssm.py
- config file: models/dssm.config

5. CDSSM
-------
this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

- model file: models/cdssm.py
- config file: models/cdssm.config

6. ARC-II
-------
this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: models/arcii.py
- model config: models/arcii.config

7. MV-LSTM
-------
this model is an implementation of <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>

- model file: models/mvlstm.py
- model config: models/mvlstm.config

8. Match-SRNN
-------
under development ....

##Citation

```
@article{fan2017matchzoo,
  title={MatchZoo: A Toolkit for Deep Text Matching},
  author={Fan, Yixing and Pang, Liang and Hou, JianPeng and Guo, Jiafeng and Lan, Yanyan and Cheng, Xueqi},
  journal={arXiv preprint arXiv:1707.07270},
  year={2017}
}
```

Project Organizers
====
- **Jiafeng Guo**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [HomePage](http://www.bigdatalab.ac.cn/~gjf/)
- **Yanyan Lan**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [HomePage](http://www.bigdatalab.ac.cn/~lanyanyan/)
- **Xueqi Cheng**
    - Institute of Computing Technolgy, Chinese Academy of Sciences 
    - [HomePage](http://www.bigdatalab.ac.cn/~cxq/)


Development Teams
====
- **Yixing Fan**
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [Google Scholar](https://scholar.google.com/citations?user=w5kGcUsAAAAJ&hl=en)
- **Liang Pang** 
    - Institute of Computing Technolgy, Chinese Academy of Sciences
    - [Google Scholar](https://scholar.google.com/citations?user=1dgQHBkAAAAJ&hl=zh-CN)
- **Jianpeng Hou** 
    - Software engineer from Google Inc.
    - [Homepage] (http://houjp.com/)

Acknowledgements
=====
We would like to express our appreciation to the following people for contributing source code to MatchZoo, including [Yixing Fan](https://scholar.google.com/citations?user=w5kGcUsAAAAJ&hl=en), [Liang Pang](https://scholar.google.com/citations?user=1dgQHBkAAAAJ&hl=zh-CN), [Liu Yang](https://sites.google.com/site/lyangwww/), [Jianpeng Hou](https://github.com/HouJP), [Zhou Yang](), [Niuguo cheng](https://github.com/niuox) etc..

