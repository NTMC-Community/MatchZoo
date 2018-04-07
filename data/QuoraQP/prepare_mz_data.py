#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)

sys.path.append('../../matchzoo/inputs/')
sys.path.append('../../matchzoo/utils/')

from preparation import Preparation
from preprocess import Preprocess, NgramUtil


def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict


def read_doc(infile):
    doc = {}
    for line in open(infile):
        r = line.strip().split()
        doc[r[0]] = r[1:]
        #assert len(doc[r[0]]) == int(r[1])
    return doc


def filter_triletter(tri_stats, min_filter_num=5, max_filter_num=10000):
    tri_dict = {}
    tri_stats = sorted(tri_stats.items(), key=lambda d:d[1], reverse=True)
    for triinfo in tri_stats:
        if triinfo[1] >= min_filter_num and triinfo[1] <= max_filter_num:
            if triinfo[0] not in tri_dict:
                tri_dict[triinfo[0]] = len(tri_dict)
    return tri_dict


if __name__ == '__main__':
    prepare = Preparation()
    srcdir = './'
    dstdir = './'

    infile = srcdir + 'quora_duplicate_questions.tsv'
    #infile = srcdir + 'train.csv'
    corpus, rels = prepare.run_with_one_corpus_for_quora(infile)
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(dstdir + 'corpus.txt', corpus)
    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, [0.8, 0.1, 0.1])
    prepare.save_relation(dstdir + 'relation_train.txt', rel_train)
    prepare.save_relation(dstdir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(dstdir + 'relation_test.txt', rel_test)
    print('Preparation finished ...')

    preprocessor = Preprocess(word_stem_config={'enable': False}, word_filter_config={'min_freq': 5})
    dids, docs = preprocessor.run(dstdir + 'corpus.txt')
    preprocessor.save_word_dict(dstdir + 'word_dict.txt')
    preprocessor.save_words_stats(dstdir + 'word_stats.txt')

    fout = open(dstdir + 'corpus_preprocessed.txt', 'w')
    for inum, did in enumerate(dids):
        fout.write('%s\t%s\n' % (did, ' '.join(map(str, docs[inum]))))
    fout.close()
    print('preprocess finished ...')
    print('preprocess finished ...')
    word_dict_input = dstdir + 'word_dict.txt'
    triletter_dict_output = dstdir + 'triletter_dict.txt'
    word_triletter_output = dstdir + 'word_triletter_map.txt'
    word_dict = read_dict(word_dict_input)
    word_triletter_map = {}
    triletter_stats = {}
    for wid, word in word_dict.items():
        nword = '#' + word + '#'
        ngrams = NgramUtil.ngrams(list(nword), 3, '')
        word_triletter_map[wid] = []
        for tric in ngrams:
            if tric not in triletter_stats:
                triletter_stats[tric] = 0
            triletter_stats[tric] += 1
            word_triletter_map[wid].append(tric)
    triletter_dict = filter_triletter(triletter_stats, 5, 10000)
    with open(triletter_dict_output, 'w') as f:
        for tri_id, tric in triletter_dict.items():
            print(tri_id, tric, file=f)
    with open(word_triletter_output, 'w') as f:
        for wid, trics in word_triletter_map.items():
            print(wid, ' '.join([str(triletter_dict[k]) for k in trics if k in triletter_dict]), file=f)

    print('Triletter Processing finished ...')
