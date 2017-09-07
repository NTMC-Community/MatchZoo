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
    basedir = '../data/example/ranking/'
    in_dict_file = basedir + 'word_dict.txt'
    in_query_file = basedir + 'corpus_preprocessed.txt'
    #in_doc_file = basedir + 'docid_doc.txt'

    out_dict_file = basedir + 'word_dict_dssm.txt'
    out_query_file = basedir + 'corpus_preprocessed_dssm.txt'
    #out_doc_file = basedir + 'did_dssm.txt'

    fout_q = open(out_query_file, 'w')
    word_dict = read_dict(in_dict_file)
    qinfo = read_doc(in_query_file)
    print 'After query read finished ...'
    new_dict = {}
    inum = 0
    for qid, qwords in qinfo.items():
        inum += 1
        qstr = '#'
        for w in qwords:
            assert w in word_dict
            qstr += word_dict[w] + '#'
        ngrams = NgramUtil.ngrams(list(qstr), 3, '')
        q_ngrams = []
        for tric in ngrams:
            if tric not in new_dict:
                new_dict[tric] = len(new_dict)
            q_ngrams.append(new_dict[tric])
        print >> fout_q, qid, len(q_ngrams), ' '.join(map(str, q_ngrams))
        if inum % 10 == 0:
            print 'inum : %d ....\r'%(inum),
    fout_q.close()
    print

    '''
    inum = 0
    fout_d = open(out_doc_file, 'w')
    dinfo = read_doc(in_doc_file)
    print 'After doc read finished ...'
    for did, dwords in dinfo.items():
        inum += 1
        dstr = '#'
        for w in dwords:
            assert w in word_dict
            dstr += word_dict[w] + '#'
        ngrams = NgramUtil.ngrams(list(dstr), 3, '')
        d_ngrams = []
        for tric in ngrams:
            if tric not in new_dict:
                new_dict[tric] = len(new_dict)
            d_ngrams.append(new_dict[tric])
        print >> fout_d, did, len(d_ngrams), ' '.join(map(str, d_ngrams))
        if inum % 10 == 0:
            print 'inum : %d ....\r'%(inum),
    fout_d.close()
    print
    '''
    with open(out_dict_file, 'w') as f:
        for ns, nid in new_dict.items():
            print >> f, ns, nid
    print 'Done ...'
