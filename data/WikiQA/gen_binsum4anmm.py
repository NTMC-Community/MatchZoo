#!/usr/bin/env python
# coding: utf-8
'''
Generate bin sum used in the attention based neural matching model (aNMM)
'''
import os
import sys
sys.path.append('../../matchzoo/utils/')
sys.path.append('../../matchzoo/inputs/')
from preprocess import cal_binsum
from rank_io import *


if __name__ == '__main__':
    bin_num = int(sys.argv[1])
    srcdir = './'
    embedfile = srcdir + 'embed_glove_d300_norm'
    corpusfile = srcdir + 'corpus_preprocessed.txt'

    relfiles = [ srcdir + 'relation_train.txt',
            srcdir + 'relation_valid.txt',
            srcdir + 'relation_test.txt'
            ]
    binfiles = [
            srcdir + 'relation_train.binsum-%d.txt' % bin_num,
            srcdir + 'relation_valid.binsum-%d.txt' % bin_num,
            srcdir + 'relation_test.binsum-%d.txt' % bin_num
            ]
    embed_dict = read_embedding(filename = embedfile)
    print('read embedding finished ...')
    _PAD_ = len(embed_dict)
    embed_size = len(embed_dict[embed_dict.keys()[0]])
    embed_dict[_PAD_] = np.zeros((embed_size, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [_PAD_+1, embed_size]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    corpus, _ = read_data(corpusfile)
    print('read corpus finished....')
    for idx, relfile in enumerate(relfiles):
        binfile = binfiles[idx]
        rel = read_relation(relfile)
        fout = open(binfile, 'w')
        for label, d1, d2 in rel:
            assert d1 in corpus
            assert d2 in corpus
            qnum = len(corpus[d1])
            d1_embed = embed[corpus[d1]]
            d2_embed = embed[corpus[d2]]
            curr_bin_sum = cal_binsum(d1_embed, d2_embed, qnum, bin_num)
            curr_bin_sum = curr_bin_sum.tolist()
            fout.write(' '.join(map(str, curr_bin_sum)))
            fout.write('\n')
        fout.close()
    print 'generate bin sum finished ...'
