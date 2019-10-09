import io
import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('matchzoo/version.py').read())

short_description = 'Facilitating the design, comparison and sharing of deep text matching models.'

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'keras >= 2.3.0',
    'nltk >= 3.2.3',
    'numpy >= 1.14',
    'tqdm >= 4.19.4',
    'dill >= 0.2.7.1',
    'pandas >= 0.23.1',
    'networkx >= 2.1',
    'h5py >= 2.8.0',
    'hyperopt >= 0.1.1'
]

extras_requires = {
    'tests': [
        'coverage >= 4.3.4',
        'codecov >= 2.0.15',
        'pytest >= 3.0.3',
        'pytest-cov >= 2.4.0',
        'flake8 >= 3.6.0',
        'flake8_docstrings >= 1.0.2'],
}


setup(
    name="MatchZoo",
    version=__version__,
    author="Yixing Fan, Bo Wang, Zeyi Wang, Liang Pang, Liu Yang, Qinghua Wang, etc.",
    author_email="fanyixing@ict.ac.cn",
    description=(short_description),
    license="Apache 2.0",
    keywords="text matching models",
    url="https://github.com/NTMC-Community/MatchZoo",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)
