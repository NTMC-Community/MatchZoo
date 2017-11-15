
import os
import sys
import numpy
import math

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    fout = open(outfile,'w')
    with open(infile,'r') as f:
        for line in f:
            r = line.split()
            w = r[0]
            vec = [float(k) for k in r[1:]]
            sum = 0.0
            for k in vec:
                sum += k*k;
            sum = math.sqrt(sum)
            for i,k in enumerate(vec):
                vec[i] /= sum;
            print >>fout, w, ' '.join(['%f'%k for k in vec])
    fout.close()
