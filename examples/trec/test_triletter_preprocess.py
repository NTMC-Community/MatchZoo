# /bin/python2.7
# -*- coding=utf8 -*-

import os
import sys
import numpy as np
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preprocess import *

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
    run_mode = 'ranking'
    if len(sys.argv) > 1 and sys.argv[1] == 'classification':
        run_mode = 'classification'
    basedir = '../../data/toy_example/%s/'%(run_mode)
    in_dict_file = basedir + 'word_dict.txt'
    out_dict_file = basedir + 'triletter_dict.txt'
    word_triletter_map_file = basedir + 'word_triletter_map.txt'

    word_dict = read_dict(in_dict_file)
    triletter_stats = {}
    word_triletter_map = {}
    for wid, word in word_dict.items():
        word_triletter_map[wid] = []
        ngrams = NgramUtil.ngrams(list('#' + word + '#'), 3, '')
        for tric in ngrams:
            if tric not in triletter_stats:
                triletter_stats[tric] = 0
            triletter_stats[tric] += 1
            word_triletter_map[wid].append(tric)
    triletter_dict = filter_triletter(triletter_stats, 1, 1000)
    with open(out_dict_file, 'w') as f:
        for triid, tric in triletter_dict.items():
            print >> f, triid, tric
    with open(word_triletter_map_file, 'w') as f:
        for wid, trics in word_triletter_map.items():
            print >> f, wid, ' '.join([str(triletter_dict[k]) for k in trics if k in triletter_dict])
    print 'Done ...'
