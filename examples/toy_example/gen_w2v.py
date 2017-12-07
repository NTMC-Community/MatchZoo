import sys
import numpy as np
from tqdm import tqdm


w2v_file = open(sys.argv[1])
word_map_file = open(sys.argv[2])

#word_count, embed_dim = w2v_file.readline().strip().split()

word_map_w2v = {}
word_dict = {}

print 'load word dict ...'
for line in tqdm(word_map_file):
  line = line.split()
  try:
      word_dict[line[0]] = int(line[1])
  except:
      print line
      continue

print 'load word vectors ...'
for line in tqdm(w2v_file):
  line = line.split()
  if len(line) == 0:
      continue
  if line[0] in word_dict:
    word_map_w2v[line[0]] = line[1:]

embed_dim = len(word_map_w2v[word_map_w2v.keys()[0]])

word_diff = list()
for w in word_dict.keys():
    if w not in word_map_w2v:
        word_diff.append(w)

# output shared w2v dict
mapped_w2v_file = open(sys.argv[3], 'w')
print 'save %d share word vectors ...' % len(word_map_w2v)
for w, vecs in tqdm(word_map_w2v.items()):
    print >> mapped_w2v_file, word_dict[w], ' '.join(vecs)

print 'save %d random word vectors ...' % len(word_diff)
for w in tqdm(word_diff):
  alpha = 0.5* (2.0 * np.random.random()  - 1.0)
  curr_embed = (2.0 * np.random.random_sample([embed_dim]) - 1.0) * alpha
  curr_embed = [ '%.6f'%k for k in curr_embed.tolist()]
  print >> mapped_w2v_file, word_dict[w], ' '.join(curr_embed)

mapped_w2v_file.close()

print 'Map word vectors finished ...'
