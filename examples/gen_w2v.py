import sys
import numpy as np


#w2v_file = open('/home/pangliang/matching/data/embedding/wikicorp.201004-pdc-ns-iter-20-alpha-0.05-window-10-dim-50-neg-10-subsample-0.0001.txt')
w2v_file = open(sys.argv[1])
word_map_file = open(sys.argv[2])

#word_count, embed_dim = w2v_file.readline().strip().split()
#word_count = int(word_count)
#embed_dim = int(embed_dim)
#word_count = 2196017
#embed_dim = 300
word_count = 400000
embed_dim = 50
print word_count, embed_dim

word_map_w2v = {}
word_map_msr = {}

for line in w2v_file:
  line = line.split()
  if len(line) == 0:
      continue
  word_map_w2v[line[0]] = line[1:]

for line in word_map_file:
  line = line.split()
  try:
      word_map_msr[line[0]] = int(line[1])
  except:
      print line
      continue

print 'word count in w2v: ', len(word_map_w2v)
print 'word count in msr: ', len(word_map_msr)

word_shared = set(word_map_w2v.keys()) & set(word_map_msr.keys())
word_diff = list(set(word_map_msr.keys()) - set(word_map_w2v.keys()))
print 'share count: ', len(word_shared)

# output small w2v dict
small_w2v_file = open(sys.argv[3], 'w')
#random_w2v_file = open(sys.argv[4], 'w')
for w in word_shared:
  print >>small_w2v_file, word_map_msr[w], ' '.join(word_map_w2v[w])

for w in word_diff:
  alpha = 0.5* (2.0 * np.random.random()  - 1.0)
  curr_embed = (2.0 * np.random.random_sample([embed_dim]) - 1.0) * alpha
  curr_embed = [ '%.6f'%k for k in curr_embed.tolist()]
  print >>small_w2v_file, word_map_msr[w], ' '.join(curr_embed)
  #print >>random_w2v_file, word_map_msr[w], ' '.join(curr_embed)

small_w2v_file.close()
#random_w2v_file.close()
