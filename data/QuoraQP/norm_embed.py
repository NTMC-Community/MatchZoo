# coding: utf8
from __future__ import print_function

import os
import sys
import codecs
import numpy
import math

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    fout = codecs.open(outfile,'w', encoding='utf8')
    with codecs.open(infile, 'r', encoding='utf8') as f:
        for line in f:
            r = line.split()
            w = r[0]
            try:
                # BUG: it will happen `name@domain.com`
                vec = [float(k) for k in r[1:]]
            except:
                print(line)
            sum = 0.0
            for k in vec:
                sum += k * k
            sum = math.sqrt(sum)
            for i,k in enumerate(vec):
                vec[i] /= sum
            print(w, ' '.join(['%f' % k for k in vec]), file=fout)
    fout.close()
