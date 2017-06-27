# -*- coding: utf8 -*-
import os
import sys
import json

from utils import *

def main(config):
    inputs = config['inputs']
    print inputs
    queries, _ = read_data(inputs['text1_corpus'])
    docs, _ = read_data(inputs['text2_corpus'])
    pair_gen = PairGenerator(data1=queries, data2=docs, config=inputs)
    return
if __name__=='__main__':
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)
        main(config)
