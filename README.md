<div align='center'>
<img src="./artworks/matchzoo-logo.png" width = "400"  alt="logo" align="center" />
</div>

# MatchZoo

> MatchZoo is a toolkit for text matching. It was developed with a focus on facilitating the designing, comparing and sharing of deep text matching models.<br/>
> MatchZoo 是一个通用的文本匹配工具包，它旨在方便大家快速的实现、比较、以及分享最新的深度文本匹配模型。

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Documentation Status](http://readthedocs.org/projects/matchzoo/badge/?version=2.0)](https://matchzoo.readthedocs.io/en/2.0/?badge=2.0)
[![Build Status](https://travis-ci.org/NTMC-Community/MatchZoo.svg?branch=2.0)](https://travis-ci.org/NTMC-Community/MatchZoo/)
[![codecov](https://codecov.io/gh/NTMC-Community/MatchZoo/branch/2.0/graph/badge.svg)](https://codecov.io/gh/NTMC-Community/MatchZoo)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Requirements Status](https://requires.io/github/NTMC-Community/MatchZoo/requirements.svg?branch=2.0)](https://requires.io/github/NTMC-Community/MatchZoo/requirements/?branch=2.0)
---

## Get Started in 60 Seconds

First, import modules and prepare input data.

```python
from matchzoo import preprocessor
from matchzoo import generators
from matchzoo import models

train = [
    ("id0", "id1", "beijing", "Beijing is capital of China", 1),
    ("id0", "id2", "beijing", "China is in east Asia", 0),
    ("id0", "id3", "beijing", "Summer in Beijing is hot.", 1)
]
test = [
    ("id0", "id4", "beijing", "I visted beijing yesterday.")
]
```

Preprocess your input data in three lines of code, keep track parameters to be passed into the model.

```python
dssm_preprocessor = preprocessor.DSSMPreprocessor()
processed_tr = dssm_preprocessor.fit_transform(train, stage='train')
processed_te = dssm_preprocessor.fit_transform(test, stage='predict')
# DSSM expect dimensionality of letter-trigrams as input shape.
# The fitted parameters has been stored in `context` during preprocessing on training data.
input_shapes = processed_tr.context['input_shapes']
```

Use MatchZoo `generators` module to generate `point-wise`, `pair-wise` or `list-wise` inputs into batches.

```python
generator_tr = generators.PointGenerator(processed_tr)
generator_te = generators.PointGenerator(processed_te)
# Example, train with generator, test with the first batch.
X_te, y_te = generator_te[0]
```

Train a [Deep Semantic Structured Model](https://www.microsoft.com/en-us/research/project/dssm/), make predictions on test data.

```python
dssm_model = models.DSSMModel()
dssm_model.params['input_shapes'] = input_shapes
dssm_model.guess_and_fill_missing_params()
dssm_model.build()
dssm_model.compile()
dssm_model.fit_generator(generator_tr)
# Make predictions
predictions = dssm_model.predict([X_te.text_left, X_te.text_right])
```

For detailed usage, such as hyper-parameters, model persistence, evaluation, please check out our [Tutorials](https://github.com/NTMC-Community/MatchZoo/tree/2.0/notebooks) or documention: [English](https://matchzoo.readthedocs.io/en/2.0/) [中文](https://matchzoo.readthedocs.io/zh/latest/)

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



## Citing MatchZoo

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

<table>
​	<tbody>
​		<tr>
​		  <td align="center" valign="top">
​			  <a href="https://github.com/faneshion"><img width="40" height="40" src="https://github.com/faneshion.png?s=40"></a><br>
​			  <a href="http://www.bigdatalab.ac.cn/~fanyixing/">Fan Yixing</a>
​			  <p> Founder </p><br>
        <p> ASST PROF, ICT</p>
​		  </td>
      <td align="center" valign="top">
​       <a href="https://github.com/pl8787"><img width="40" height="40" src="https://github.com/pl8787.png?s=40"></a><br>
​       <a href="https://github.com/pl8787">Pang Liang</a>
​       <p> Core Dev</p><br>
       <p> ASST PROF, ICT</p>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/yangliuy"><img width="40" height="40" src="https://github.com/yangliuy.png?s=40"></a><br>
​       <a href="https://github.com/yangliuy">Yang Liu</a>
       <p> Core Dev</p><br>
​       <p> PhD. UMASS</p>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/bwanglzu"><img width="40" height="40" src="https://github.com/bwanglzu.png?s=40"></a><br>
​       <a href="https://github.com/bwanglzu">Wang Bo</a>
​       <p> Core Dev</p><br>
       <p> M.S. TU Delft </p>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/uduse"><img width="36" height="36" src="https://github.com/uduse.png?s=36"></a><br>
​       <a href="https://github.com/uduse">Wang Zeyi</a>
       <p> Core Dev</p><br>
​       <p> B.S. UC Davis</p>
​      </td>
    </tr><tr>
      <td align="center" valign="top">
​       <a href="https://github.com/wqh17101"><img width="40" height="40" src="https://github.com/wqh17101.png?s=40"></a><br>
​       <a href="https://github.com/wqh17101">Wang Qinghua</a>
​       <p> Documentation </p><br>
       <p> B.S. Shandong Univ.</p>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/ZizhenWang"><img width="40" height="40" src="https://github.com/ZizhenWang.png?s=40"></a><br>
​       <a href="https://github.com/ZizhenWang">Wang Zizhen</a>
​       <p> Dev</p><br>
       <p> M.S. UCAS</p>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/lixinsu"><img width="40" height="40" src="https://github.com/lixinsu.png?s=40"></a><br>
​       <a href="https://github.com/lixinsu">Su Lixin</a>
       <p> Dev</p><br>
​       <p> PhD. UCAS</p><br>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/zhouzhouyang520"><img width="40" height="40" src="https://github.com/zhouzhouyang520.png?s=40"></a><br>
​       <a href="https://github.com/zhouzhouyang520">Yang Zhou</a>
​       <p> Dev</p><br>
       <p> M.S. CQUT</p>
​      </td>
      <td align="center" valign="top">
​       <a href="https://github.com/rgtjf"><img width="36" height="36" src="https://github.com/rgtjf.png?s=36"></a><br>
​       <a href="https://github.com/rgtjf">Tian Junfeng</a>
​       <p> Dev</p><br>
       <p> M.S. ECNU</p>
​      </td>
​		</tr>
​	</tbody>
</table>



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


## Contribution

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before creating a pull request. If you have a MatchZoo-related paper/project/compnent/tool, send a pull request to [this awesome list](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match)!


## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2015-present, Yixing Fan (faneshion)