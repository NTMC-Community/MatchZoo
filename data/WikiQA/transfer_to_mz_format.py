#!/usr/bin/env python
# coding: utf-8

import os
import sys


basedir = './WikiQACorpus/'
dstdir = './'
infiles = [ basedir + 'WikiQA-train.txt', basedir + 'WikiQA-dev-filtered.txt', basedir + 'WikiQA-test-filtered.txt' ]
outfiles = [ dstdir + 'WikiQA-mz-train.txt', dstdir + 'WikiQA-mz-dev.txt', dstdir + 'WikiQA-mz-test.txt' ]

for idx,infile in enumerate(infiles):
    outfile = outfiles[idx]
    fout = open(outfile, 'w')
    for line in open(infile, 'r'):
        r = line.strip().split('\t')
        fout.write('%s\t%s\t%s\n'%(r[2], r[0], r[1]))
    fout.close()



