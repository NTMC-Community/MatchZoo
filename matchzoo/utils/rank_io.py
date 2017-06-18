#! encoding: utf-8
#! author: pangliang

import json
import numpy as np

# Read Word Dict and Inverse Word Dict
def read_word_dict(filename):
    word_dict = {}
    iword_dict = {}
    for line in open(filename):
        line = line.strip().split()
        word_dict[int(line[1])] = line[0]
        iword_dict[line[0]] = int(line[1])
    print '[%s]\n\tWord dict size: %d' % (filename, len(word_dict))
    return word_dict, iword_dict

# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = map(float, line[1:])
    print '[%s]\n\tEmbedding size: %d' % (filename, len(embed))
    return embed

# Read old version data
def read_data_old_version(filename):
    data = []
    for idx, line in enumerate(open(filename)):
        line = line.strip().split()
        len1 = int(line[1])
        len2 = int(line[2])
        data.append([map(int, line[3:3+len1]), map(int, line[3+len1:])])
        assert len2 == len(data[idx][1])
    print '[%s]\n\tInstance size: %d' % (filename, len(data))
    return data

# Read Relation Data
def read_relation(filename):
    data = []
    for line in open(filename):
        line = line.strip().split()
        data.append( (int(line[0]), line[1], line[2]) )
    print '[%s]\n\tInstance size: %s' % (filename, len(data))
    return data

# Read Data Dict
def read_data(filename):
    data = {}
    for line in open(filename):
        line = line.strip().split()
        data[line[0]] = map(int, line[2:])
    print '[%s]\n\tData size: %s' % (filename, len(data))
    return data

# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[embed_dict.keys()[0]])
    if embed is None:
        embed = np.zeros( (feat_size, max_size), dtype = np.float32 )
    for k in embed_dict:
        embed[k] = np.array(embed_dict[k])
    print 'Generate numpy embed:', embed.shape
    return embed

