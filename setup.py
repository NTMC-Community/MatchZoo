import io
import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('matchzoo/version.py').read())

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'keras >= 2.0.5',
    'tensorflow >= 1.1.0',
    'nltk >= 3.2.3',
    'numpy >= 1.12.1',
    'six >= 1.10.0',
    'h5py >= 2.7.0',
    'tqdm >= 4.19.4',
    'scipy >= 1.0.0',
    'jieba >= 0.39',
    'psutil >= 5.4.6'
]

extras_requires = {
    'visualize': ['matplotlib >= 2.2.0'],
    'tests': [
        'coverage >= 4.3.4',
        'codecov >= 2.0.15',
        'pytest >= 3.0.3',
        'pytest-cov >= 2.4.0',
        'mock >= 2.0.0',
        'flake8 >= 3.2.1',
        'flake8_docstrings >= 1.0.2'],
}


setup(
    name="MatchZoo",
    version=__version__,
    author="Yixing Fan, Liang Pang, Jianpeng Hou, Jiafeng Guo, Yanyan Lan, Xueqi Cheng",
    author_email="fanyixing@software.ict.ac.cn",
    description=("MatchZoo is a toolkit for text matching. It was developed with a focus on facilitating the designing, comparing and sharing of deep text matching models."),
    license="Apache 2.0",
    keywords="text matching models",
    url="https://github.com/faneshion/MatchZoo",
    packages=find_packages(),
    long_description=long_description,
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)
