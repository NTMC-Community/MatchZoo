"""
Parse the log files of MatchZoo to draw the learning curves of
deep learning model training process to observe the training loss
and metrics on train/dev/test data. With these curves, we can do
a better job in model debugging
@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@author: Thiziri Belkacem (belkacemthiziri@gmail.com)
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

def draw_train_learning_curve(log_file):
    r = open(log_file, 'r')
    info = r.readlines()
    r.close() 
    start_line = '[Model] Model Compile Done.'
    start_flag = False
    train_loss = []
    valid_map = []
    valid_p10 = []
    test_map = []
    test_p10 = []
    for line in info: 
        line = line.strip('\r\n')
        if start_flag:
            # print (line)
            tokens = line.split('\t')
            if len(tokens) > 1:
                # print('tokens: ', tokens)
                if 'train' in line:
                    # print(tokens)
                    train_loss.append(float(tokens[2].split('=')[1]))
                elif 'valid' in line:
                    # print (tokens)
                    if len(tokens) < 6:
                        continue
                    map_token = [token for token in tokens if "map" in token]
                    print("map\t", map_token)
                    p10_token = [token for token in tokens if "precision@10" in token]
                    print("p10\t", p10_token)
                    valid_map.append(float(map_token[0].split('=')[1]))
                    valid_p10.append(float(p10_token[0].split('=')[1]))
                else:
                    # print(tokens[2], tokens[5])
                    map_token = [token for token in tokens if "map" in token]
                    print("map\t", map_token)
                    p10_token = [token for token in tokens if "precision@10" in token]
                    print("p10\t", p10_token)
                    test_map.append(float(map_token[0].split('=')[1]))
                    test_p10.append(float(p10_token[0].split('=')[1]))
        if start_line in line.strip('\r\n'):
            start_flag = True

    # draw learning curve
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    x = range(len(train_loss))
    # print ('x', x)
    # print (train_loss)
    if (len(train_loss) != len(test_p10)) or (len(train_loss) != len(valid_p10)):
        print('wait a few seconds for the valid/test metrics in the next iteration...')
        min_len = min(len(test_p10), len(valid_p10))
        train_loss = train_loss[0:min_len]
        valid_map = valid_map[0:min_len]
        valid_p10 = valid_p10[0:min_len]
        test_map = test_map[0:min_len]
        test_p10 = test_p10[0:min_len]
        x = x[0:min_len]
    line1, = plt.plot(x, train_loss, 'r-v', label='train_loss')
    line2, = plt.plot(x, valid_map, 'c-s', label='valid_map')
    line3, = plt.plot(x, valid_p10, 'm-o', label='valid_p10')
    line4, = plt.plot(x, test_map, 'b-+', label='test_map')
    line5, = plt.plot(x, test_p10, 'g-^', label='test_p10')
    plt.legend(handles=[line1, line2, line3, line4, line5], loc=1)
    rcParams['grid.linestyle'] = 'dotted'
    plt.grid()
    min_loss = min(train_loss)
    max_p10 = max(test_p10)
    max_map = max(test_map)
    plt.ylabel('Model training curves')
    log_label = """Best performances map = {map} in iterations {it_map}, P@10 = {p10} in iterations {it_p10} and 
    loss = {los} in iterations {it_los}""".format(map=max_map,
                                                  it_map=str([idx+1 for idx, val in enumerate(test_map) if val == max_map]),
                                                  p10=max_p10,
                                                  it_p10=str([idx+1 for idx, val in enumerate(test_p10) if val == max_p10]),
                                                  los=min_loss,
                                                  it_los=str([idx+1 for idx, val in enumerate(train_loss) if val == min_loss]))
    plt.title(log_label)
    plt.xlabel('learning iterations')
    plt.show()
    # plt.savefig("_".join(log_label.split())+".png")


exp_id = 1
print('Exp ', exp_id, ': model comparation')
if not os.path.isfile(sys.argv[1]):
    for model in os.listdir(sys.argv[1]):
        log_file = os.path.join(sys.argv[1], model)  # 'path_of_log_file'
        log_label = model + ' training'
        draw_train_learning_curve(log_file, log_label)
else:
    log_file = sys.argv[1]  # file.log containing training details
    draw_train_learning_curve(log_file)
