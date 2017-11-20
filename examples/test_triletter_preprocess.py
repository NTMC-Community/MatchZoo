# /bin/python2.7
# -*- coding=utf8 -*-

import os
import sys
import numpy as np
sys.path.append('../matchzoo/inputs')
sys.path.append('../matchzoo/utils')
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

if __name__ == '__main__':
    basedir = '../data/example/classification/'
    in_dict_file = basedir + 'word_dict.txt'
    out_dict_file = basedir + 'triletter_dict.txt'
    word_triletter_map_file = basedir + 'word_triletter_map.txt'

    word_dict = read_dict(in_dict_file)
    triletter_dict = {}
    word_triletter_map = {}
    for wid, word in word_dict.items():
        word_triletter_map[wid] = []
        ngrams = NgramUtil.ngrams(list('#' + word + '#'), 3, '')
        for tric in ngrams:
            if tric not in triletter_dict:
                triletter_dict[tric] = len(triletter_dict)
            word_triletter_map[wid].append(triletter_dict[tric])
    with open(out_dict_file, 'w') as f:
        for triid, tric in triletter_dict.items():
            print >> f, triid, tric
    with open(word_triletter_map_file, 'w') as f:
        for wid, trics in word_triletter_map.items():
            print >> f, wid, ' '.join(map(str,trics))
    print 'Done ...'
