# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys


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
    def __init__(self, text1_file, text2_file):
        self.word_dict = {}
        self.word_stats = {}
        self.text1 = {}
        self.text2 = {}
        return
    
    def run(self):
        return 

    def remap(self, text1_file):
        return

    def rare_word_remove(self):
        return

    def frequency_word_remove(self):
        return

    def stop_word_remove(self, stopword=[]):
        return 

    def stem(self):
        return

    @worddict.setter
    def worddict(self, worddict):
        self.word_dict = worddict

    def save_worddict(self, file_path):
        return

    def save_text1(self, file_path):
        return

    def save_text2(self, file_path):
        return
