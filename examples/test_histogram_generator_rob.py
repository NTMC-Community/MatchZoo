# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('../matchzoo/inputs')
sys.path.append('../matchzoo/utils')
from preprocess import *
from rank_io import *


if __name__ == '__main__':
    hist_size = 30
    path = '../data/robust/'
    embedfile = path + 'embed_wv_d300_norm'
    queryfile = path + 'qid_query.txt'
    docfile = path + 'docid_doc.txt'
    relfile = path + 'relation.test.fold1.txt'
    histfile = path + 'relation.test.fold1.hist-%d.txt'%(hist_size)

    # note here word embeddings have been normalized to speed up calculation
    embed_dict = read_embedding(filename = embedfile) 
    print('after read embedding ...')
    _PAD_ = 162502 # for word without wordembeeding, assign an random embedding
    embed_dict[_PAD_] = np.zeros((300, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.02, 0.02, [162503, 300]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    query, _ = read_data(queryfile)
    print('after read query ....')
    doc, _ = read_data(docfile)
    print('after read doc ...')
    rel = read_relation(relfile)
    print('after read relation ... ')
    fout = open(histfile, 'w')
    inum = 0
    for label, d1, d2 in rel:
        inum += 1
        assert d1 in query
        assert d2 in doc
        qnum = len(query[d1])
        d1_embed = embed[query[d1]]
        d2_embed = embed[doc[d2]]
        curr_hist = cal_hist(d1_embed, d2_embed, qnum, hist_size)
        curr_hist = curr_hist.tolist()
        fout.write(' '.join(map(str, curr_hist)))
        fout.write('\n')
        if inum % 1000 == 0:
            print('inum: %d ....\r'%inum,)
            sys.stdout.flush()
        #print(curr_hist)
    fout.close()
    print('\nfinished ...')
