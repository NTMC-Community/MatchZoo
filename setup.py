import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding= 'utf').read()

setup(
    name = "MatchZoo",
    version = "0.2.0",
    author = "Yixing Fan, Liang Pang, Jianpeng Hou, Jiafeng Guo, Yanyan Lan, Xueqi Cheng",
    author_email = "fanyixing@software.ict.ac.cn",
    description = ("MatchZoo is a toolkit for text matching. It was developed with a focus on facilitating the designing, comparing and sharing of deep text matching models."),
    license = "BSD",
    keywords = "text matching models",
    url = "https://github.com/faneshion/MatchZoo",
    packages=find_packages(),#['data', 'docs', 'examples', 'matchzoo', 'tests'],
    #long_description=read('README.md'),
    classifiers=[
        # How mature is this project? Common values are
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache License",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=[
      'keras >= 2.0.5',
      'tensorflow >= 1.1.0',
      'nltk >= 3.2.3',
      'numpy >= 1.12.1',
      'six >= 1.10.0',
      'h5py >= 2.7.0',
      'tqdm >= 4.19.4'
    ]
)
