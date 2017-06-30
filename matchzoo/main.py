# -*- coding: utf8 -*-
import os
import sys
import json
import argparse

from collections import OrderedDict

import keras
import keras.backend as K
from keras.models import Sequential, Model

from utils import *
from inputs import *
from metrics import *
from losses import *

def train(config):
    # read basic config
    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    weights_file = global_conf['weights_file']
    num_batch = global_conf['num_batch']

    model = Model.from_config(config['model'])
    rank_eval = rank_evaluations.rank_eval(rel_threshold = 0.)

    loss = []
    for lobj in config['losses']:
      loss.append(rank_losses.get(lobj))
    metrics = []
    for mobj in config['metrics']:
        metrics.append(mobj)
    model.compile(optimizer=optimizer, loss=loss)
    print '[Model] Model Compile Done.'

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # list all input tags and construct tags config
    tag = input_conf.keys()
    tag.remove('share')
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for t in tag:
        if input_conf[t]['phase'] == 'TRAIN':
            input_train_conf[t] = {}
            input_train_conf[t].update(share_input_conf)
            input_train_conf[t].update(input_conf[t])
        elif input_conf[t]['phase'] == 'EVAL':
            input_eval_conf[t] = {}
            input_eval_conf[t].update(share_input_conf)
            input_eval_conf[t].update(input_conf[t])
    print '[Input] Process %d Input Tags. %s.' % (len(tag), tag)

    # collect dataset identification
    dataset = {}
    for t in input_conf:
        if 'text1_corpus' in input_conf[t]:
            datapath = input_conf[t]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
        if 'text2_corpus' in input_conf[t]:
            datapath = input_conf[t]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
    print '[Dataset] %s Dataset Load Done.' % len(dataset)

    # initial data generator
    train_gen = OrderedDict()
    train_genfun = OrderedDict()
    eval_gen = OrderedDict()
    eval_genfun = OrderedDict()

    for tag, conf in input_train_conf.items():
        print conf
        train_gen[tag] = PairGenerator( data1 = dataset[conf['text1_corpus']],
                                      data2 = dataset[conf['text2_corpus']],
                                      config = conf )
        train_genfun[tag] = train_gen[tag].get_batch_generator()

    for tag, conf in input_eval_conf.items():
        print conf
        eval_gen[tag] = ListGenerator( data1 = dataset[conf['text1_corpus']],
                                     data2 = dataset[conf['text2_corpus']],
                                     config = conf )  
        eval_genfun[tag] = eval_gen[tag].get_batch_generator()



    for i_e in range(global_conf['num_epochs']):
        print '[Train] @ %s epoch.' % i_e
        for tag, genfun in train_genfun.items():
            print '[Train] @ %s' % tag
            model.fit_generator(
                    genfun,
                    steps_per_epoch = num_batch,
                    epochs = 1,
                    verbose = 1
                ) #callbacks=[eval_map])
        res = dict([[k,0.] for k in metrics])
        
        for tag, genfun in eval_genfun.items():
            print '[Eval] @ %s' % tag
            num_valid = 0
            for input_data, y_true in genfun:
                y_pred = model.predict(input_data)
                curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred, metrics=metrics)
                for k, v in curr_res.items():
                    res[k] += v
                num_valid += 1
            print 'epoch: %d,' %( i_e ), '  '.join(['%s:%f'%(k,v/num_valid) for k, v in res.items()]), ' ...'
            sys.stdout.flush()
            eval_genfun[tag] = eval_gen[tag].get_batch_generator()

    model.save_weights(weights_file)
    return

def predict(config):
    input_conf = config['inputs']
    print input_conf
    queries, _ = read_data(input_conf['text1_corpus'])
    docs, _ = read_data(input_conf['text2_corpus'])
    list_gen = ListGenerator(data1=queries, data2=docs, config=input_conf)

    global_conf = config["global"]
    weights_file = global_conf['weights_file']

    model = Model.from_config(config['model'])
    model.load_weights(weights_file)
    rank_eval = rank_evaluations.rank_eval(rel_threshold = 0.)

    metrics = []
    for mobj in config['metrics']:
        metrics.append(mobj)
    res = dict([[k,0.] for k in metrics])
    num_valid = 0
    res_scores = {} 
    for (x1, x1_len, x2, x2_len, y_true, id_pairs) in list_gen.get_batch:
        y_pred = model.predict({'query': x1, 'doc': x2})
        curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred, metrics=metrics)
        for k,v in curr_res.items():
            res[k] += v
        y_pred = np.squeeze(y_pred)
        for p, y in zip(id_pairs, y_pred):
            if p[0] not in res_scores:
                res_scores[p[0]] = {}
            res_scores[p[0]][p[1]] = y
        num_valid += 1
    if config['outputs']['save_test_format'] == 'trec':
        with open(config['outputs']['save_test_file'], 'w') as f:
            for qid, dinfo in res_scores.items():
                dinfo = sorted(dinfo.items(), key=lambda d:d[1], reverse=True)
                for inum,(did, score) in enumerate(dinfo):
                    print >> f, '%s\tQ0\t%s\t%d\t%f\t%s'%(qid, did, inum, score, config['net_name'])
    print 'Predict results: ', '  '.join(['%s:%f'%(k,v/num_valid) for k, v in res.items()]), ' ...'
    #print 'epoch: %d, batch : %d , map: %f, ndcg@3: %f, ndcg@5: %f ...'%(k, i, res[0], res[1], res[2])
    sys.stdout.flush()
    return

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/matchzoo.model', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    if args.phase == 'train':
        train(config)
    else:
        predict(config)
    return

if __name__=='__main__':
    main(sys.argv)
