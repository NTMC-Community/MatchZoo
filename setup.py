import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding= 'utf').read()

setup(
    name = "MatchZoo",
    version = "1.0",
    author = "Yixing Fan, Liang Pang, Jianpeng Hou, Jiafeng Guo, Yanyan Lan, Xueqi Cheng",
    author_email = "fanyixing@software.ict.ac.cn",
    description = ("MatchingZoom is a toolkit for text matching."
                    "It was developed with a focus on enabling fast experimentation."),
    license = "BSD",
    keywords = "text matching models",
    url = "https://github.com/faneshion/MatchZoo",
    packages=find_packages(),#['data', 'docs', 'examples', 'matchzoo', 'tests'],
    #long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: BSD License"],
    install_requires=[
      'keras >= 2.0.5',
      'tensorflow >= 1.1.0',
      'nltk >= 3.2.3',
      'numpy >= 1.12.1',
      'six >= 1.10.0',
      'tqdm >= 4.19.4']
)
