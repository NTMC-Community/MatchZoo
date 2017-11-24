#!/usr/bin/env python
# coding: utf-8


import os
import sys
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

if __name__ == '__main__':
    prepare = Preparation()
    srcdir = './'
    dstdir = './'

    infiles = [ srcdir + 'WikiQA-mz-train.txt', srcdir + 'WikiQA-mz-dev.txt', srcdir + 'WikiQA-mz-test.txt']
    corpus, rel_train, rel_valid, rel_test = prepare.run_with_train_valid_test_corpus(infiles[0], infiles[1], infiles[2])
    print('total corpus : %d ...' % (len(corpus)))
    print('total relation-train : %d ...' % (len(rel_train)))
    print('total relation-valid : %d ...' % (len(rel_valid)))
    print('total relation-test: %d ...' % (len(rel_test)))
    prepare.save_corpus(dstdir + 'corpus.txt', corpus)

    prepare.save_relation(dstdir + 'relation_train.txt', rel_train)
    prepare.save_relation(dstdir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(dstdir + 'relation_test.txt', rel_test)
    print('Preparation finished ...')

    preprocessor = Preprocess(word_stem_config={'enable': False}, word_filter_config={'min_freq': 2})
    dids, docs = preprocessor.run(dstdir + 'corpus.txt')
    preprocessor.save_word_dict(dstdir + 'word_dict.txt', True)
    preprocessor.save_words_stats(dstdir + 'word_stats.txt', True)

    fout = open(dstdir + 'corpus_preprocessed.txt', 'w')
    for inum, did in enumerate(dids):
        fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
    fout.close()
    print('Preprocess finished ...')

    #dssm_corp_input = dstdir + 'corpus_preprocessed.txt'
    #dssm_corp_output = dstdir + 'corpus_preprocessed_dssm.txt'
    word_dict_input = dstdir + 'word_dict.txt'
    triletter_dict_output = dstdir + 'triletter_dict.txt'
    word_triletter_output = dstdir + 'word_triletter_map.txt'
    word_dict = read_dict(word_dict_input)
    triletter_dict = {}
    word_triletter_map = {}
    for wid, word in word_dict.items():
        nword = '#' + word + '#'
        ngrams = NgramUtil.ngrams(list(nword), 3, '')
        word_triletter_map[wid] = []
        for tric in ngrams:
            if tric not in triletter_dict:
                triletter_dict[tric] = len(triletter_dict)
            word_triletter_map[wid].append(triletter_dict[tric])
    with open(triletter_dict_output, 'w') as f:
        for tri_id, tric in triletter_dict.items():
            print >> f, tri_id, tric
    with open(word_triletter_output, 'w') as f:
        for wid, tri_ids in word_triletter_map.items():
            print >> f, wid, ' '.join(map(str, tri_ids))

    print('Triletter Processing finished ...')

