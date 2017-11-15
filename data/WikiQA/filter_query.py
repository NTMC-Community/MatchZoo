#!/usr/bin/env python
# coding: utf-8

import os
import sys


if __name__ == '__main__':
    basedir = './WikiQACorpus/'
    filter_reffile = [basedir + 'WikiQA-dev-filtered.ref', basedir + 'WikiQA-test-filtered.ref']
    in_reffile = [basedir + 'WikiQA-dev.ref', basedir + 'WikiQA-test.ref']
    in_corpfile = [basedir + 'WikiQA-dev.txt', basedir + 'WikiQA-test.txt']
    outfile = [basedir + 'WikiQA-dev-filtered.txt', basedir + 'WikiQA-test-filtered.txt']

    for i in range(len(filter_reffile)):
        fout = open(outfile[i], 'w')

        filtered_qids = set()
        for line in open(filter_reffile[i], 'r'):
            r = line.strip().split()
            filtered_qids.add(r[0])

        all_qids = []
        for line in open(in_reffile[i], 'r'):
            r = line.strip().split()
            all_qids.append(r[0])


        for idx,line in enumerate(open(in_corpfile[i], 'r')):
            if all_qids[idx] not in filtered_qids:
                continue
            print >> fout, line.strip()
        fout.close()

