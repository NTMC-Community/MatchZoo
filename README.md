<div align='center'>
<img src="./artworks/matchzoo-logo.png" width = "400"  alt="logo" align="center" />
</div>

# MatchZoo

> Facilitating the design, comparison and sharing of deep text matching models.<br/>
> MatchZoo 是一个通用的文本匹配工具包，它旨在方便大家快速的实现、比较、以及分享最新的深度文本匹配模型。

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Documentation Status](https://readthedocs.org/projects/matchzoo/badge/?version=master)](https://matchzoo.readthedocs.io/en/master/?badge=master)
[![Build Status](https://travis-ci.org/NTMC-Community/MatchZoo.svg?branch=master)](https://travis-ci.org/NTMC-Community/MatchZoo/)
[![codecov](https://codecov.io/gh/NTMC-Community/MatchZoo/branch/master/graph/badge.svg)](https://codecov.io/gh/NTMC-Community/MatchZoo)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Requirements Status](https://requires.io/github/NTMC-Community/MatchZoo/requirements.svg?branch=master)](https://requires.io/github/NTMC-Community/MatchZoo/requirements/?branch=master)
---

## Get Started in 60 Seconds

To train a [Deep Semantic Structured Model](https://www.microsoft.com/en-us/research/project/dssm/), import matchzoo and prepare input data.

```python
import matchzoo as mz

train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')
predict_pack = mz.datasets.wiki_qa.load_data('test', task='ranking')
```

Preprocess your input data in three lines of code, keep track parameters to be passed into the model.

```python
preprocessor = mz.preprocessors.DSSMPreprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack)
valid_pack_processed = preprocessor.transform(valid_pack)
predict_pack_processed = preprocessor.transform(predict_pack)
```

Make use of MatchZoo customized loss functions and evaluation metrics:

```python
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
```

Initialize the model, fine-tune the hyper-parameters.

```python
model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.guess_and_fill_missing_params()
model.build()
model.compile()
```

Generate pair-wise training data on-the-fly, evaluate model performance using customized callbacks on prediction data.

```python
train_generator = mz.PairDataGenerator(train_pack_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)

pred_x, pred_y = predict_pack_processed[:].unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_x))

history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)
```

For detailed usage, such as hyper-parameters tunning, model persistence, evaluation, please check out our [tutorials](https://github.com/NTMC-Community/MatchZoo/tree/2.0/tutorials) and documention: [English](https://matchzoo.readthedocs.io/en/2.0/) [中文](https://matchzoo.readthedocs.io/zh/latest/)

If you're interested in the cutting-edge research progress, please take a look at [awaresome neural models for semantic match](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match).

## Install

MatchZoo is dependent on [Keras](https://github.com/keras-team/keras), please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend. Two ways to install MatchZoo:

**Install MatchZoo from Pypi:**

```python
pip install matchzoo
```

**Install MatchZoo from the Github source:**

```python
git clone https://github.com/NTMC-Community/MatchZoo.git
cd MatchZoo
python setup.py install
```



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
        <p>Founder<br>
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