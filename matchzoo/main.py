# -*- coding: utf8 -*-
import os
import sys
import json
import argparse

import keras
import keras.backend as K
from keras.models import Sequential, Model

from utils import *
from metrics import *
from losses import *

def train(config):
    input_conf = config['inputs']
    print input_conf
    queries, _ = read_data(input_conf['text1_corpus'])
    docs, _ = read_data(input_conf['text2_corpus'])
    pair_gen = PairGenerator(data1=queries, data2=docs, config=input_conf)
    list_gen = ListGenerator(data1=queries, data2=docs, config=input_conf)

    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    weights_file = global_conf['weights_file']

    model = Model.from_config(config['model'])
    rank_eval = rank_evaluations.rank_eval(rel_threshold = 0.)

    loss = []
    for lobj in config['losses']:
      loss.append(rank_losses.get(lobj))
    #model.compile(optimizer=keras.optimizers.Adam(lr=global_conf['learning_rate']), loss=rank_losses.get(config['losses'][0]))
    model.compile(optimizer=optimizer, loss=loss)

    for k in range(global_conf['num_epochs']):
        num_batch = pair_gen.num_pairs / input_conf['batch_size']
        for i in range(num_batch):
            x1, x1_len, x2, x2_len, y = pair_gen.get_batch
            model.fit({'query':x1, 'doc':x2}, y, batch_size=input_conf['batch_size']*2,
                    epochs = 1,
                    verbose = 1
                    ) #callbacks=[eval_map])
            if i % 100 == 0:
                res = [0., 0., 0.] 
                num_valid = 0
                for (x1, x1_len, x2, x2_len, y_true) in list_gen.get_batch:
                    y_pred = model.predict({'query': x1, 'doc': x2})
                    curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred, metrics=['map', 'ndcg@3', 'ndcg@5'])
                    res[0] += curr_res['map']
                    res[1] += curr_res['ndcg@3']
                    res[2] += curr_res['ndcg@5']
                    num_valid += 1
                res[0] /= num_valid
                res[1] /= num_valid
                res[2] /= num_valid
                print 'epoch: %d, batch : %d , map: %f, ndcg@3: %f, ndcg@5: %f ...'%(k, i, res[0], res[1], res[2])
                sys.stdout.flush()
                list_gen.reset
    model.save_weights(weights_file)
    return
def test(config):
    input_conf = config['inputs']
    print input_conf
    queries, _ = read_data(input_conf['text1_corpus'])
    docs, _ = read_data(input_conf['text2_corpus'])
    list_gen = ListGenerator(data1=queries, data2=docs, config=input_conf)

    global_conf = config["global"]
    weights_file = global_conf['weights_file']
    save_trec_file = global_conf['save_trec_file']

    model = Model.from_config(config['model'])
    model.load_weights(weights_file)
    rank_eval = rank_evaluations.rank_eval(rel_threshold = 0.)

    res = [0., 0., 0.] 
    num_valid = 0
    for (x1, x1_len, x2, x2_len, y_true) in list_gen.get_batch:
        y_pred = model.predict({'query': x1, 'doc': x2})
        curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred, metrics=['map', 'ndcg@3', 'ndcg@5'])
        res[0] += curr_res['map']
        res[1] += curr_res['ndcg@3']
        res[2] += curr_res['ndcg@5']
        num_valid += 1
    res[0] /= num_valid
    res[1] /= num_valid
    res[2] /= num_valid
    print 'epoch: %d, batch : %d , map: %f, ndcg@3: %f, ndcg@5: %f ...'%(k, i, res[0], res[1], res[2])
    sys.stdout.flush()
    return

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--model_file', default=False, help='MatchZoo model file for the chosen model')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    if args.phase == 'train':
        train(config)
    else:
        test(config)
    return

if __name__=='__main__':
    '''
    if len(sys.argv) < 3:
        help()
        exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--model_file', default=False, help='MatchZoo model file for the chosen model')
    args = parser.parse_args()
    '''
    main(sys.argv)
    '''
    with open(config_file, 'r') as f:
        config = json.load(f)
        main(config)
    '''
