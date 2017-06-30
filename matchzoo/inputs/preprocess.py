# -*- coding: utf-8 -*-


from __future__ import print_function
from nltk.tokenize import word_tokenize
import jieba
import sys
from nltk.stem import SnowballStemmer


class Preprocess(object):
    """Abstract preprocess class
    # Properties
        word_dict:
        rword_dict:
        word_stats:
        text1:
        text2:
        relation:
    # Methods
        rare_word_remove(): remove rare words(e.g. high-frequency words and low-frequency words)
        run(): 
    """

    _valid_lang = ['en', 'cn']
    _stemmer = SnowballStemmer('english')

    def __init__(self, lang='en',
                 stop_words=list(),
                 min_freq=1,
                 max_freq=sys.maxint,
                 min_len=1,
                 max_len=sys.maxint,
                 word_dict=None):
        assert lang.lower() in Preprocess._valid_lang, 'Wrong language type: %s' % lang
        self._lang = lang
        self._stop_words = stop_words
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._min_len = min_len
        self._max_len = max_len
        self._word_dict = word_dict
    
    def run(self, file_path):
        dids, docs = Preprocess.load(file_path)
        docs = Preprocess.word_seg(docs, self._lang)
        docs = Preprocess.word_stem(docs)
        dids, docs = Preprocess.word_filter(docs,
                                            dids=dids,
                                            stop_words=self._stop_words,
                                            min_freq=self._min_freq,
                                            max_freq=self._max_freq,
                                            min_len=self._min_len,
                                            max_len=self._max_len)
        docs, self._word_dict = Preprocess.word_index(docs, word_dict=self._word_dict)
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
        f = open(file_path, 'r')
        for line in f:
            line = line.strip()
            if '' != line:
                did, doc = Preprocess.parse(line)
                dids.append(did)
                docs.append(doc)
        return dids, docs

    @staticmethod
    def word_seg_en(docs):
        docs = [word_tokenize(sent) for sent in docs]
        return docs

    @staticmethod
    def word_seg_cn(docs):
        docs = [list(jieba.cut(sent)) for sent in docs]
        return docs

    @staticmethod
    def word_seg(docs, lang):
        assert lang.lower() in Preprocess._valid_lang, 'Wrong language type: %s' % lang
        docs = getattr(Preprocess, '%s_%s' % (sys._getframe().f_code.co_name, lang))(docs)
        return docs

    @staticmethod
    def cal_doc_freq(docs):
        wdf = dict()
        for ws in docs:
            ws = set(ws)
            for w in ws:
                wdf[w] = wdf.get(w, 0) + 1
        return wdf

    @staticmethod
    def word_filter(docs, dids=None, stop_words=list(), min_freq=1, max_freq=sys.maxint, min_len=1, max_len=sys.maxint):
        ws_filter = set()
        # filter with stop_words
        ws_filter.update(stop_words)
        # filter with min_freq and max_freq
        wdf = Preprocess.cal_doc_freq(docs)
        for w in wdf:
            if min_freq > wdf[w] or max_freq < wdf[w]:
                ws_filter.add(w)
        # filter with min_len and max_len
        if dids is None:
            docs = [ws for ws in docs if min_len <= len(ws) <= max_len]
        else:
            new_docs = list()
            new_dids = list()
            for i in range(len(docs)):
                if min_len <= len(docs[i]) <= max_len:
                    new_docs.append(docs[i])
                    new_dids.append(dids[i])
            docs = new_docs
            dids = new_dids
        # filter with illegal words
        docs = [[w for w in ws if w not in ws_filter] for ws in docs]
        if dids is None:
            return docs
        else:
            return dids, docs

    @staticmethod
    def word_stem(docs):
        docs = [[Preprocess._stemmer.stem(w) for w in ws] for ws in docs]
        return docs

    @staticmethod
    def build_word_dict(docs):
        word_dict = dict()
        for ws in docs:
            for w in ws:
                word_dict.setdefault(w, len(word_dict))
        return word_dict

    @staticmethod
    def word_index(docs, word_dict=None):
        if word_dict is None:
            word_dict = Preprocess.build_word_dict(docs)
        docs = [[word_dict[w] for w in ws if w in word_dict] for ws in docs]
        return docs, word_dict


def _test():
    file_path = '/Users/houjianpeng/tmp/txt'
    # dids, docs = Preprocess.load(file_path)
    # docs = Preprocess.word_seg_en(docs)
    # docs = Preprocess.word_seg_cn(docs)
    # docs = Preprocess.word_seg(docs, 'en')
    # dids, docs = Preprocess.word_filter(docs, dids=dids)
    # docs = Preprocess.word_stem(docs)
    # docs, word_dict = Preprocess.word_index(docs)
    # print(dids)
    # print(docs)
    # print(word_dict)
    preprocessor = Preprocess()
    dids, docs = preprocessor.run(file_path)
    print(dids)
    print(docs)
    print(preprocessor._word_dict)


if __name__ == '__main__':
    _test()
