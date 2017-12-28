# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import codecs
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

    def get_text_id(self, hashid, text, idtag='T'):
        hash_obj = hashlib.sha1(text.encode('utf8'))  # if the text are the same, then the hash_code are also the same
        hex_dig = hash_obj.hexdigest()
        if hex_dig in hashid:
            return hashid[hex_dig]
        else:
            tid = idtag + str(len(hashid))  # start from 0, 1, 2, ...
            hashid[hex_dig] = tid
            return tid

    def parse_line(self, line, delimiter='\t'):
        subs = line.split(delimiter)
        # print('subs: ', len(subs))
        if 3 != len(subs):
            raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        else:
            return subs[0], subs[1], subs[2]

    def run_with_one_corpus(self, file_path):
        hashid = {}
        corpus = {}
        rels = []
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in f:
            line = line
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
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in f:
            line = line
            line = line.strip()
            label, t1, t2 = self.parse_line(line)
            id1 = self.get_text_id(hashid_q, t1, 'Q')
            id2 = self.get_text_id(hashid_d, t2, 'D')
            corpus_q[id1] = t1
            corpus_d[id2] = t2
            rels.append((label, id1, id2))
        f.close()
        return corpus_q, corpus_d, rels

    def run_with_train_valid_test_corpus(self, train_file, valid_file, test_file):
        '''
        Run with pre-splited train_file, valid_file, test_file
        The input format should be label \t text1 \t text2
        The query ids can't be duplicated. For the same query
        id, the document ids can't be duplicated.
        Note that if we make queries with unique id (fixed 10 candidates for a single query), then it is
        possible that multiple queries have different query ids, but with the same text (in rare cases)
        :param train_file: train file
        :param valid_file: valid file
        :param test_file: test file
        :return: corpus, rels_train, rels_valid, rels_test
        '''
        hashid = {}
        corpus = {}
        rels = []
        rels_train = []
        rels_valid = []
        rels_test = []
        # merge corpus files, but return rels for train/valid/test seperately
        curQ = 'init'
        curQid = 0
        for file_path in list([train_file, valid_file, test_file]):
            if file_path == train_file:
                rels = rels_train
            elif file_path == valid_file:
                rels = rels_valid
            if file_path == test_file:
                rels = rels_test
            f = codecs.open(file_path, 'r', encoding='utf8')
            for line in f:
                line = line
                line = line.strip()
                label, t1, t2 = self.parse_line(line)
                id2 = self.get_text_id(hashid, t2, 'D')
                # generate unique query ids
                if t1 == curQ:
                    # same query
                    id1 = 'Q' + str(curQid)
                else:
                    # new query
                    curQid += 1
                    id1 = 'Q' + str(curQid)
                    curQ = t1
                corpus[id1] = t1
                corpus[id2] = t2
                rels.append((label, id1, id2))
            f.close()
        return corpus, rels_train, rels_valid, rels_test

    @staticmethod
    def save_corpus(file_path, corpus):
        f = codecs.open(file_path, 'w', encoding='utf8')
        for qid, text in corpus.items():
            f.write('%s %s\n' % (qid, text))
        f.close()

    @staticmethod
    def merge_corpus(train_corpus, valid_corpus, test_corpus):
        # cat train valid test > corpus.txt
        # cat corpus_train.txt corpus_valid.txt corpus_test.txt > corpus.txt
        os.system('cat ' + train_corpus + ' ' + valid_corpus + ' ' + test_corpus + '  > corpus.txt')

    @staticmethod
    def save_relation(file_path, relations):
        f = open(file_path, 'w')
        for rel in relations:
            f.write('%s %s %s\n' % (rel))
        f.close()

    @staticmethod
    def check_filter_query_with_dup_doc(input_file):
        """ Filter queries with duplicated doc ids in the relation files
        :param input_file: input file, which could be the relation file for train/valid/test data
                           The format is "label qid did"
        :return:
        """
        with open(input_file) as f_in, open(input_file + '.fd', 'w') as f_out:
            cur_qid = 'init'
            cache_did_set = set()
            cache_q_lines = []
            found_dup_doc = False
            for l in f_in:
                tokens = l.split()
                if tokens[1] == cur_qid:
                    # same qid
                    cache_q_lines.append(l)
                    if tokens[2] in cache_did_set:
                        found_dup_doc = True
                    else:
                        cache_did_set.add(tokens[2])
                else:
                    # new qid
                    if not found_dup_doc:
                        f_out.write(''.join(cache_q_lines))
                    else:
                        print('found qid with duplicated doc id/text: ', ''.join(cache_q_lines))
                        print('filtered... continue')
                    cache_q_lines = []
                    cache_q_lines.append(l)
                    found_dup_doc = False
                    cache_did_set.clear()
                    cur_qid = tokens[1]
                    cache_did_set.add(tokens[2])
            # the last query
            # print len(cache_q_lines), len(cache_did_set)
            if len(cache_q_lines) != 0 and len(cache_q_lines) == len(cache_did_set):
                f_out.write(''.join(cache_q_lines))
                print('write the last query... done: ', ''.join(cache_q_lines))

    @staticmethod
    def split_train_valid_test(relations, ratio=(0.8, 0.1, 0.1)):
        random.shuffle(relations)
        total_rel = len(relations)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid
        rel_train = relations[: num_train]
        rel_valid = relations[num_train: valid_end]
        rel_test = relations[valid_end:]
        return rel_train, rel_valid, rel_test

    @staticmethod
    def split_train_valid_test_for_ranking(relations, ratio=(0.8, 0.1, 0.1)):
        qid_group = set()
        for r, q, d in relations:
            qid_group.add(q)
        qid_group = list(qid_group)

        random.shuffle(qid_group)
        total_rel = len(qid_group)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid

        qid_train = qid_group[: num_train]
        qid_valid = qid_group[num_train: valid_end]
        qid_test = qid_group[valid_end:]

        def select_rel_by_qids(qids):
            rels = []
            qids = set(qids)
            for r, q, d in relations:
                if q in qids:
                    rels.append((r, q, d))
            return rels

        rel_train = select_rel_by_qids(qid_train)
        rel_valid = select_rel_by_qids(qid_valid)
        rel_test = select_rel_by_qids(qid_test)

        return rel_train, rel_valid, rel_test


if __name__ == '__main__':
    prepare = Preparation()
    basedir = '../../data/example/ranking/'
    corpus, rels = prepare.run_with_one_corpus(basedir + 'sample.txt')
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(basedir + 'corpus.txt', corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, (0.8, 0.1, 0.1))
    prepare.save_relation(basedir + 'relation_train.txt', rel_train)
    prepare.save_relation(basedir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(basedir + 'relation_test.txt', rel_test)
    print('Done ...')
