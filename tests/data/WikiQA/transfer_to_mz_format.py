# coding: utf-8

import os
import sys
import hashlib

def get_text_id(hashid, text, idtag='T'):
    hash_obj = hashlib.sha1(text) #.encode('utf8'))  # if the text are the same, then the hash_code are also the same
    hex_dig = hash_obj.hexdigest()
    if hex_dig in hashid:
        return hashid[hex_dig]
    else:
        tid = idtag + str(len(hashid))  # start from 0, 1, 2, ...
        hashid[hex_dig] = tid
        return tid

basedir = './WikiQACorpus/'
dstdir = './'
infiles = [ basedir + 'WikiQA-train.txt', basedir + 'WikiQA-dev-filtered.txt', basedir + 'WikiQA-test-filtered.txt' ]
outfiles = [ dstdir + 'WikiQA-mz-train.txt', dstdir + 'WikiQA-mz-dev.txt', dstdir + 'WikiQA-mz-test.txt' ]

qid = 0
did = 0
hashid = {}
for idx, infile in enumerate(infiles):
    records = {}
    for line in open(infile):
        r = line.strip().split('\t')
        assert len(r) == 3
        if r[0] not in records:
            records[r[0]] = []
        records[r[0]].append((r[1], r[2]))
    outfile = outfiles[idx]
    fout = open(outfile, 'w')
    for q, info in records.items():
        cqid = 'qid' + str(qid)
        for idx, dinfo in enumerate(info):
            #cdid = 'did' + str(idx)
            cdid = get_text_id(hashid, dinfo[0], 'did')
            #print('%s\t%s\t%s\t%s\t%s'%(cqid, cdid, q, dinfo[0], dinfo[1]), file=fout)
            fout.write('%s\t%s\t%s\t%s\t%s\n'%(cqid, cdid, q, dinfo[0], dinfo[1]))
            #print('%s\t%s\t%s\t%s'%(cqid, cdid, q, dinfo[0]), file=fout)
        qid += 1
    fout.close()
