# -*- coding: utf-8 -*-


from __future__ import print_function
import sys
import numpy as np
import hashlib
import random


import preprocess


class Preparation(object):
    '''Convert dataset of different text matching tasks into a unified format as the input of deep matching modules. Users provide datasets contain pairs of texts along with their labels, and the module produces the following files:   
    * Word Dictionary: this file records the mapping from each word to a unique identifier.
    * Corpus File: this file records the mapping from each text to a unique identifiers, along with a sequence of word identifiers contained in text.
    * Relation File: this file records the relationship between two texts, each line containing the label and a pair of ids.
    '''

    def __init__(self):
        pass

    def get_text_id(self, hashid, text, idtag = 'T'):
        hash_obj = hashlib.sha1(text.encode('utf8'))
        hex_dig = hash_obj.hexdigest()
        if hex_dig in hashid:
            return hashid[hex_dig]
        else:
            tid = idtag + str(len(hashid))
            hashid[hex_dig] = tid
            return tid

    def parse_line(self, line, delimiter='\t'):
        subs = line.split(delimiter)
        if 3 != len(subs):
            raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        else:
            return subs[0], subs[1], subs[2]

    def run_with_one_corpus(self, file_path):
        hashid = {}
        corpus = {}
        rels = []
        f = open(file_path, 'r')
        for line in f:
            line = line.decode('utf8')
            line = line.strip()
            label, t1, t2 = self.parse_line(line)
            id1 = self.get_text_id(hashid, t1, 'T')
            id2 = self.get_text_id(hashid, t2, 'T')
            corpus[id1] = t1
            corpus[id2] = t2
            rels.append((label, id1, id2))
        f.close()
        return corpus, rels
    
    def run_with_two_corpus(self, file_path):
        hashid_q = {}
        hashid_d = {}
        corpus_q = {}
        corpus_d = {}
        rels = []
        f = open(file_path, 'r')
        for line in f:
            line = line.decode('utf8')
            line = line.strip()
            label, t1, t2 = self.parse_line(line)
            id1 = self.get_text_id(hashid_q, t1, 'Q')
            id2 = self.get_text_id(hashid_d, t2, 'D')
            corpus_q[id1] = t1
            corpus_d[id2] = t2
            rels.append((label, id1, id2))
        f.close()
        return corpus_q, corpus_d, rels

    @staticmethod
    def save_corpus(file_path, corpus):
        f = open(file_path, 'w')
        for qid, text in corpus.items():
            f.write('%s %s\n' % (qid, text.encode('utf8')))
        f.close()

    @staticmethod
    def save_relation(file_path, relations):
        f = open(file_path, 'w')
        for rel in relations:
            f.write('%s %s %s\n' % (rel))
        f.close()

    @staticmethod
    def split_train_valid_test(relations, ratio=[0.8, 0.1, 0.1]):
        random.shuffle(relations)
        total_rel = len(relations)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid
        rel_train = relations[: num_train]
        rel_valid = relations[num_train: valid_end]
        rel_test = relations[valid_end:]
        return rel_train, rel_valid, rel_test

if __name__ == '__main__':
    prepare = Preparation()
    basedir = '../../data/example/'
    corpus, rels = prepare.run_with_one_corpus(basedir + 'sample_train.txt')
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(basedir + 'corpus.txt', corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, [0.8, 0.1, 0.1])
    prepare.save_relation(basedir + 'relation_train.txt', rel_train)
    prepare.save_relation(basedir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(basedir + 'relation_test.txt', rel_test)
    print('Done ...')
