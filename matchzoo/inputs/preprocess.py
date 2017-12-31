# -*- coding: utf-8 -*-
from __future__ import print_function

import jieba
import sys
import six
import codecs
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

sys.path.append('../inputs')
sys.path.append('../utils')
from preparation import *
from rank_io import *


class Preprocess(object):

    _valid_lang = ['en', 'cn']
    _stemmer = SnowballStemmer('english')

    def __init__(self,
                 word_seg_config = {},
                 doc_filter_config = {},
                 word_stem_config = {},
                 word_lower_config = {},
                 word_filter_config = {},
                 word_index_config = {}
                 ):
        # set default configuration
        self._word_seg_config = { 'enable': True, 'lang': 'en' }
        self._doc_filter_config = { 'enable': True, 'min_len': 0, 'max_len': six.MAXSIZE }
        self._word_stem_config = { 'enable': True }
        self._word_lower_config = { 'enable': True }
        self._word_filter_config = { 'enable': True, 'stop_words': nltk_stopwords.words('english'),
                                     'min_freq': 1, 'max_freq': six.MAXSIZE, 'words_useless': None }
        self._word_index_config = { 'word_dict': None }

        self._word_seg_config.update(word_seg_config)
        self._doc_filter_config.update(doc_filter_config)
        self._word_stem_config.update(word_stem_config)
        self._word_lower_config.update(word_lower_config)
        self._word_filter_config.update(word_filter_config)
        self._word_index_config.update(word_index_config)

        self._word_dict = self._word_index_config['word_dict']
        self._words_stats = dict()

    def run(self, file_path):
        print('load...')
        dids, docs = Preprocess.load(file_path)

        if self._word_seg_config['enable']:
            print('word_seg...')
            docs = Preprocess.word_seg(docs, self._word_seg_config)

        if self._doc_filter_config['enable']:
            print('doc_filter...')
            dids, docs = Preprocess.doc_filter(dids, docs, self._doc_filter_config)

        if self._word_stem_config['enable']:
            print('word_stem...')
            docs = Preprocess.word_stem(docs)

        if self._word_lower_config['enable']:
            print('word_lower...')
            docs = Preprocess.word_lower(docs)

        self._words_stats = Preprocess.cal_words_stat(docs)

        if self._word_filter_config['enable']:
            print('word_filter...')
            docs, self._words_useless = Preprocess.word_filter(docs, self._word_filter_config, self._words_stats)

        print('word_index...')
        docs, self._word_dict = Preprocess.word_index(docs, self._word_index_config)

        return dids, docs

    @staticmethod
    def parse(line):
        subs = line.split(' ', 1)
        if 1 == len(subs):
            return subs[0], ''
        else:
            return subs[0], subs[1]

    @staticmethod
    def load(file_path):
        dids = list()
        docs = list()
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in tqdm(f):
            line = line.strip()
            if '' != line:
                did, doc = Preprocess.parse(line)
                dids.append(did)
                docs.append(doc)
        f.close()
        return dids, docs

    @staticmethod
    def word_seg_en(docs):
        docs = [word_tokenize(sent) for sent in tqdm(docs)]
        # show the progress of word segmentation with tqdm
        '''docs_seg = []
        print('docs size', len(docs))
        for i in tqdm(range(len(docs))):
            docs_seg.append(word_tokenize(docs[i]))'''
        return docs

    @staticmethod
    def word_seg_cn(docs):
        docs = [list(jieba.cut(sent)) for sent in docs]
        return docs

    @staticmethod
    def word_seg(docs, config):
        assert config['lang'].lower() in Preprocess._valid_lang, 'Wrong language type: %s' % config['lang']
        docs = getattr(Preprocess, '%s_%s' % (sys._getframe().f_code.co_name, config['lang']))(docs)
        return docs

    @staticmethod
    def cal_words_stat(docs):
        words_stats = {}
        docs_num = len(docs)
        for ws in docs:
            for w in ws:
                if w not in words_stats:
                    words_stats[w] = {}
                    words_stats[w]['cf'] = 0
                    words_stats[w]['df'] = 0
                    words_stats[w]['idf'] = 0
                words_stats[w]['cf'] += 1
            for w in set(ws):
                words_stats[w]['df'] += 1
        for w, winfo in words_stats.items():
            words_stats[w]['idf'] = np.log( (1. + docs_num) / (1. + winfo['df']))
        return words_stats

    @staticmethod
    def word_filter(docs, config, words_stats):
        if config['words_useless'] is None:
            config['words_useless'] = set()
            # filter with stop_words
            config['words_useless'].update(config['stop_words'])
            # filter with min_freq and max_freq
            for w, winfo in words_stats.items():
                # filter too frequent words or rare words
                if config['min_freq'] > winfo['df'] or config['max_freq'] < winfo['df']:
                    config['words_useless'].add(w)
        # filter with useless words
        docs = [[w for w in ws if w not in config['words_useless']] for ws in tqdm(docs)]
        return docs, config['words_useless']

    @staticmethod
    def doc_filter(dids, docs, config):
        new_docs = list()
        new_dids = list()
        for i in tqdm(range(len(docs))):
            if config['min_len'] <= len(docs[i]) <= config['max_len']:
                new_docs.append(docs[i])
                new_dids.append(dids[i])
        return new_dids, new_docs

    @staticmethod
    def word_stem(docs):
        docs = [[Preprocess._stemmer.stem(w) for w in ws] for ws in tqdm(docs)]
        return docs

    @staticmethod
    def word_lower(docs):
        docs = [[w.lower() for w in ws] for ws in tqdm(docs)]
        return docs

    @staticmethod
    def build_word_dict(docs):
        word_dict = dict()
        for ws in docs:
            for w in ws:
                word_dict.setdefault(w, len(word_dict))
        return word_dict

    @staticmethod
    def word_index(docs, config):
        if config['word_dict'] is None:
            config['word_dict'] = Preprocess.build_word_dict(docs)
        docs = [[config['word_dict'][w] for w in ws if w in config['word_dict']] for ws in tqdm(docs)]
        return docs, config['word_dict']

    @staticmethod
    def save_lines(file_path, lines):
        f = codecs.open(file_path, 'w', encoding='utf8')
        for line in lines:
            line = line
            f.write(line + "\n")
        f.close()

    @staticmethod
    def load_lines(file_path):
        f = codecs.open(file_path, 'r', encoding='utf8')
        lines = f.readlines()
        f.close()
        return lines

    @staticmethod
    def save_dict(file_path, dic, sort=False):
        if sort:
            dic = sorted(dic.items(), key=lambda d:d[1], reverse=False)
            lines = ['%s %s' % (k, v) for k, v in dic]
        else:
            lines = ['%s %s' % (k, v) for k, v in dic.items()]
        Preprocess.save_lines(file_path, lines)

    @staticmethod
    def load_dict(file_path):
        lines = Preprocess.load_lines(file_path)
        dic = dict()
        for line in lines:
            k, v = line.split()
            dic[k] = v
        return dic

    def save_words_useless(self, words_useless_fp):
        Preprocess.save_lines(words_useless_fp, self._words_useless)

    def load_words_useless(self, words_useless_fp):
        self._words_useless = set(Preprocess.load_lines(words_useless_fp))

    def save_word_dict(self, word_dict_fp, sort=False):
        Preprocess.save_dict(word_dict_fp, self._word_dict, sort)

    def load_word_dict(self, word_dict_fp):
        self._word_dict = Preprocess.load_dict(word_dict_fp)

    def save_words_stats(self, words_stats_fp, sort=False):
        if sort:
            word_dic = sorted(self._word_dict.items(), key=lambda d:d[1], reverse=False)
            lines = ['%s %d %d %f' % (wid, self._words_stats[w]['cf'], self._words_stats[w]['df'],
                self._words_stats[w]['idf']) for w, wid in word_dic]
        else:
            lines = ['%s %d %d %f' % (wid, self._words_stats[w]['cf'], self._words_stats[w]['df'],
                self._words_stats[w]['idf']) for w, wid in self._word_dict.items()]
        Preprocess.save_lines(words_stats_fp, lines)

    def load_words_stats(self, words_stats_fp):
        lines = Preprocess.load_lines(words_stats_fp)
        for line in lines:
            wid, cf, df, idf  = line.split()
            self._words_stats[wid] = {}
            self._words_stats[wid]['cf'] = int(cf)
            self._words_stats[wid]['df'] = int(df)
            self._words_stats[wid]['idf'] = float(idf)


class NgramUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def unigrams(words):
        """
            Input: a list of words, e.g., ["I", "am", "Denny"]
            Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([words[i], words[i + k]]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([words[i], words[i + k1], words[i + k1 + k2]]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def uniterms(words):
        return NgramUtil.unigrams(words)

    @staticmethod
    def biterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for j in range(i + 1, L):
                    lst.append(join_string.join([words[i], words[j]]))
        else:
            # set it as uniterm
            lst = NgramUtil.uniterms(words)
        return lst

    @staticmethod
    def triterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in xrange(L - 2):
                for j in xrange(i + 1, L - 1):
                    for k in xrange(j + 1, L):
                        lst.append(join_string.join([words[i], words[j], words[k]]))
        else:
            # set it as biterm
            lst = NgramUtil.biterms(words, join_string)
        return lst

    @staticmethod
    def fourterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
            Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                for j in xrange(i + 1, L - 2):
                    for k in xrange(j + 1, L - 1):
                        for l in xrange(k + 1, L):
                            lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
        else:
            # set it as triterm
            lst = NgramUtil.triterms(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram

    @staticmethod
    def nterms(words, nterm, join_string=" "):
        """wrapper for nterm"""
        if nterm == 1:
            return NgramUtil.uniterms(words)
        elif nterm == 2:
            return NgramUtil.biterms(words, join_string)
        elif nterm == 3:
            return NgramUtil.triterms(words, join_string)
        elif nterm == 4:
            return NgramUtil.fourterms(words, join_string)

def cal_hist(t1_rep, t2_rep, qnum, hist_size):
    #qnum = len(t1_rep)
    mhist = np.zeros((qnum, hist_size), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))
    for (i,j), v in np.ndenumerate(mm):
        if i >= qnum:
            break
        vid = int((v + 1.) / 2. * (hist_size - 1.))
        mhist[i][vid] += 1.
    mhist += 1.
    mhist = np.log10(mhist)
    return mhist.flatten()

def cal_binsum(t1_rep, t2_rep, qnum, bin_num):
    mbinsum = np.zeros((qnum, bin_num), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))
    for (i, j), v in np.ndenumerate(mm):
        if i >= qnum:
            break
        vid = int((v + 1.) / 2. * (bin_num - 1.))
        mbinsum[i][vid] += v
    #mhist += 1. # smooth is not needed for computing bin sum
    #mhist = np.log10(mhist) # not needed for computing  bin sum
    return mbinsum.flatten()

def _test_ngram():
    words = 'hello, world! hello, deep!'
    print(NgramUtil.ngrams(list(words), 3, ''))

def _test_hist():
    embedfile = '../../data/mq2007/embed_wiki-pdc_d50_norm'
    queryfile = '../../data/mq2007/qid_query.txt'
    docfile = '../../data/mq2007/docid_doc.txt'
    relfile = '../../data/mq2007/relation.test.fold5.txt'
    histfile = '../../data/mq2007/relation.test.fold5.hist-30.txt'
    embed_dict = read_embedding(filename = embedfile)
    print('after read embedding ...')
    _PAD_ = 193367
    embed_dict[_PAD_] = np.zeros((50, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [193368, 50]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    query, _ = read_data(queryfile)
    print('after read query ....')
    doc, _ = read_data(docfile)
    print('after read doc ...')
    rel = read_relation(relfile)
    print('after read relation ... ')
    fout = open(histfile, 'w')
    for label, d1, d2 in rel:
        assert d1 in query
        assert d2 in doc
        qnum = len(query[d1])
        d1_embed = embed[query[d1]]
        d2_embed = embed[doc[d2]]
        curr_hist = cal_hist(d1_embed, d2_embed, qnum, 30)
        curr_hist = curr_hist.tolist()
        fout.write(' '.join(map(str, curr_hist)))
        fout.write('\n')
        print(qnum)
        #print(curr_hist)
    fout.close()



if __name__ == '__main__':
    #_test_ngram()
    # test with sample data
    basedir = '../../data/example/ranking/'
    prepare = Preparation()
    sample_file = basedir + 'sample.txt'
    corpus, rels = prepare.run_with_one_corpus(sample_file)
    print ('total corpus size', len(corpus))
    print ('total relations size', len(rels))
    prepare.save_corpus(basedir + 'corpus.txt', corpus)
    prepare.save_relation(basedir + 'relation.txt', rels)
    print ('preparation finished ...')

    print ('begin preprocess...')
    # Prerpocess corpus file
    preprocessor = Preprocess(min_freq=1)
    dids, docs = preprocessor.run(basedir + 'corpus.txt')
    preprocessor.save_word_dict(basedir + 'word_dict.txt')
    preprocessor.save_words_stats(basedir + 'word_stats.txt')

    fout = open(basedir + 'corpus_preprocessed.txt', 'w')
    for inum, did in enumerate(dids):
        fout.write('%s\t%s\n' % (did, ' '.join(map(str, docs[inum]))))
    fout.close()
    print('preprocess finished ...')


