<div align='center'>
<img src="https://github.com/NTMC-Community/MatchZoo/blob/master/artworks/matchzoo-logo.png?raw=true" width = "400"  alt="logo" align="center" />
</div>

# MatchZoo [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=MatchZoo:%20deep%20learning%20for%20semantic%20matching&url=https://github.com/NTMC-Community/MatchZoo)

> Facilitating the design, comparison and sharing of deep text matching models.<br/>
> MatchZoo 是一个通用的文本匹配工具包，它旨在方便大家快速的实现、比较、以及分享最新的深度文本匹配模型。

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/NTMC-Community/community)
[![Pypi Downloads](https://img.shields.io/pypi/dm/matchzoo.svg?label=pypi)](https://pypi.org/project/MatchZoo/)
[![Documentation Status](https://readthedocs.org/projects/matchzoo/badge/?version=master)](https://matchzoo.readthedocs.io/en/master/?badge=master)
[![Build Status](https://travis-ci.org/NTMC-Community/MatchZoo.svg?branch=master)](https://travis-ci.org/NTMC-Community/MatchZoo/)
[![codecov](https://codecov.io/gh/NTMC-Community/MatchZoo/branch/master/graph/badge.svg)](https://codecov.io/gh/NTMC-Community/MatchZoo)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Requirements Status](https://requires.io/github/NTMC-Community/MatchZoo/requirements.svg?branch=master)](https://requires.io/github/NTMC-Community/MatchZoo/requirements/?branch=master)
---

The goal of MatchZoo is to provide a high-quality codebase for deep text matching research, such as document retrieval, question answering, conversational response ranking, and paraphrase identification. With the unified data processing pipeline, simplified model configuration and automatic hyper-parameters tunning features equipped, MatchZoo is flexible and easy to use.

<table>
  <tr>
    <th width=30%, bgcolor=#999999 >Tasks</th> 
    <th width=20%, bgcolor=#999999>Text 1</th>
    <th width="20%", bgcolor=#999999>Text 2</th>
    <th width="20%", bgcolor=#999999>Objective</th>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Paraphrase Indentification </td>
    <td align="center", bgcolor=#eeeeee> string 1 </td>
    <td align="center", bgcolor=#eeeeee> string 2 </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Textual Entailment </td>
    <td align="center", bgcolor=#eeeeee> text </td>
    <td align="center", bgcolor=#eeeeee> hypothesis </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Question Answer </td>
    <td align="center", bgcolor=#eeeeee> question </td>
    <td align="center", bgcolor=#eeeeee> answer </td>
    <td align="center", bgcolor=#eeeeee> classification/ranking </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Conversation </td>
    <td align="center", bgcolor=#eeeeee> dialog </td>
    <td align="center", bgcolor=#eeeeee> response </td>
    <td align="center", bgcolor=#eeeeee> classification/ranking </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Information Retrieval </td>
    <td align="center", bgcolor=#eeeeee> query </td>
    <td align="center", bgcolor=#eeeeee> document </td>
    <td align="center", bgcolor=#eeeeee> ranking </td>
  </tr>
</table>

## Get Started in 60 Seconds

To train a [Deep Semantic Structured Model](https://www.microsoft.com/en-us/research/project/dssm/), import matchzoo and prepare input data.

```python
import matchzoo as mz

train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')
```

Preprocess your input data in three lines of code, keep track parameters to be passed into the model.

```python
preprocessor = mz.preprocessors.DSSMPreprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
```

Make use of MatchZoo customized loss functions and evaluation metrics:

```python
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]
```

Initialize the model, fine-tune the hyper-parameters.

```python
model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.guess_and_fill_missing_params()
model.build()
model.compile()
```

Generate pair-wise training data on-the-fly, evaluate model performance using customized callbacks on validation data.

```python
train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)
valid_x, valid_y = valid_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_x))
history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)
```

## References
[Tutorials](https://github.com/NTMC-Community/MatchZoo/tree/master/tutorials)

[English Documentation](https://matchzoo.readthedocs.io/en/master/)

[中文文档](https://matchzoo.readthedocs.io/zh/latest/)

If you're interested in the cutting-edge research progress, please take a look at [awaresome neural models for semantic match](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match).

## Install

MatchZoo is dependent on [Keras](https://github.com/keras-team/keras), please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend. Two ways to install MatchZoo:

**Install MatchZoo from Pypi:**

```python
pip install matchzoo
```

**Install MatchZoo from the Github source:**

```
git clone https://github.com/NTMC-Community/MatchZoo.git
cd MatchZoo
python setup.py install
```


## Models:

1. [DRMM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/drmm.py): this model is an implementation of <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.

2. [MatchPyramid](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/match_pyramid.py): this model is an implementation of <a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>

3. [ARC-I](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/arci.py): this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

4. [DSSM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/dssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

5. [CDSSM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/cdssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

6. [ARC-II](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/arcii.py): this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

7. [MV-LSTM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/mvlstm.py):this model is an implementation of <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>

8. [aNMM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/anmm.py): this model is an implementation of <a href="http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>

9. [DUET](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/duet.py): this model is an implementation of <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>

10. [K-NRM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/knrm.py): this model is an implementation of <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a>

11. [CONV-KNRM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/conv_knrm.py): this model is an implementation of <a href="http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf">Convolutional neural networks for soft-matching n-grams in ad-hoc search</a>

12. models under development: <a href="https://arxiv.org/abs/1604.04378">Match-SRNN</a>, <a href="https://arxiv.org/abs/1710.05649">DeepRank</a>, <a href="https://arxiv.org/abs/1702.03814">BiMPM</a> .... 


## Citation

If you use MatchZoo in your research, please use the following BibTex entry.

```
@article{fan2017matchzoo,
  title={Matchzoo: A toolkit for deep text matching},
  author={Fan, Yixing and Pang, Liang and Hou, JianPeng and Guo, Jiafeng and Lan, Yanyan and Cheng, Xueqi},
  journal={arXiv preprint arXiv:1707.07270},
  year={2017}
}
```


## Development Team

​ ​ ​ ​ ​
<table border="0">
  <tbody>
    <tr align="center">
      <td>
        ​ <a href="https://github.com/faneshion"><img width="40" height="40" src="https://github.com/faneshion.png?s=40" alt="faneshion"></a><br>
        ​ <a href="http://www.bigdatalab.ac.cn/~fanyixing/">Fan Yixing</a> ​
        <p>Core Dev<br>
        ASST PROF, ICT</p>​
      </td>
      <td>
         <a href="https://github.com/bwanglzu"><img width="40" height="40" src="https://github.com/bwanglzu.png?s=40" alt="bwanglzu"></a><br>
         <a href="https://github.com/bwanglzu">Wang Bo</a> ​
        <p>Core Dev<br> M.S. TU Delft</p>​
      </td>
      <td>
        ​ <a href="https://github.com/uduse"><img width="40" height="40" src="https://github.com/uduse.png?s=36" alt="uduse"></a><br>
         <a href="https://github.com/uduse">Wang Zeyi</a>
         <p>Core Dev<br> B.S. UC Davis</p>​
      </td>
      <td>
        ​ <a href="https://github.com/pl8787"><img width="40" height="40" src="https://github.com/pl8787.png?s=40" alt="pl8787"></a><br>
        ​ <a href="https://github.com/pl8787">Pang Liang</a>
        <p>Core Dev<br>
        ASST PROF, ICT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/yangliuy"><img width="40" height="40" src="https://github.com/yangliuy.png?s=40" alt="yangliuy"></a><br>
        ​ <a href="https://github.com/yangliuy">Yang Liu</a>
        <p>Core Dev<br>
        PhD. UMASS</p>​
      </td>
    </tr>
    <tr align="center">
      <td>
        ​ <a href="https://github.com/wqh17101"><img width="40" height="40" src="https://github.com/wqh17101.png?s=40" alt="wqh17101"></a><br>
        ​ <a href="https://github.com/wqh17101">Wang Qinghua</a> ​
        <p>Documentation<br>
        B.S. Shandong Univ.</p>​
      </td>
      <td>
        ​ <a href="https://github.com/ZizhenWang"><img width="40" height="40" src="https://github.com/ZizhenWang.png?s=40" alt="ZizhenWang"></a><br>
        ​ <a href="https://github.com/ZizhenWang">Wang Zizhen</a> ​
        <p>Dev<br>
        M.S. UCAS</p>​
      </td>
      <td>
        ​ <a href="https://github.com/lixinsu"><img width="40" height="40" src="https://github.com/lixinsu.png?s=40" alt="lixinsu"></a><br>
        ​ <a href="https://github.com/lixinsu">Su Lixin</a>
        <p>Dev<br>
        PhD. UCAS</p>​
      </td>
      <td>
        ​ <a href="https://github.com/zhouzhouyang520"><img width="40" height="40" src="https://github.com/zhouzhouyang520.png?s=40" alt="zhouzhouyang520"></a><br>
        ​ <a href="https://github.com/zhouzhouyang520">Yang Zhou</a> ​
        <p>Dev<br>
        M.S. CQUT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/rgtjf"><img width="40" height="40" src="https://github.com/rgtjf.png?s=36" alt="rgtjf"></a><br>
        ​ <a href="https://github.com/rgtjf">Tian Junfeng</a> ​
        <p>Dev<br>
        M.S. ECNU</p>​
      </td>
    </tr>
  </tbody>
</table>



## Contribution

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before creating a pull request. If you have a MatchZoo-related paper/project/compnent/tool, send a pull request to [this awesome list](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match)!

Thank you to all the people who already contributed to MatchZoo!

[Jianpeng Hou](https://github.com/HouJP), [Lijuan Chen](https://github.com/githubclj), [Yukun Zheng](https://github.com/zhengyk11), [Niuguo Cheng](https://github.com/niuox), [Dai Zhuyun](https://github.com/AdeDZY), [Aneesh Joshi](https://github.com/aneesh-joshi), [Zeno Gantner](https://github.com/zenogantner), [Kai Huang](https://github.com/hkvision), [stanpcf](https://github.com/stanpcf), [ChangQF](https://github.com/ChangQF), [Mike Kellogg
](https://github.com/wordreference)




## Project Organizers

- Jiafeng Guo
  * Institute of Computing Technology, Chinese Academy of Sciences
  * [Homepage](http://www.bigdatalab.ac.cn/~gjf/)
- Yanyan Lan
  * Institute of Computing Technology, Chinese Academy of Sciences
  * [Homepage](http://www.bigdatalab.ac.cn/~lanyanyan/)
- Xueqi Cheng
  * Institute of Computing Technology, Chinese Academy of Sciences
  * [Homepage](http://www.bigdatalab.ac.cn/~cxq/)


## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2015-present, Yixing Fan (faneshion)
