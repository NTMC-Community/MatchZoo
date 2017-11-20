# -*- coding: utf8 -*-
#drow the ROC
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
# Force matplotlib to not use any Xwindows backend.

def calculate_roc_auc(file_path, roc_save_path, title):
    db = []
    pos_count, neg_count = 0, 0
    with open(file_path, 'r') as fs:
        for line in fs:
            _, _, _, _, score, _, label = line.strip().split('\t')
            score = float(score)
            label = int(label)
            db.append([score, label])
            if label == 0:
                neg_count += 1
            else:
                pos_count += 1
    db = sorted(db, key=lambda x:x[0], reverse=True) #é™åº
    #è®¡ç®—ROCåæ ‡ç‚¹
    xy_arr = []
    tp, fp = 0., 0.
    for i in range(len(db)):
        if db[i][1] == 0:
            fp += 1
        else:
            tp += 1
        xy_arr.append([fp/neg_count, tp/pos_count])
    #è®¡ç®—æ›²çº¿ä¸‹é¢ç§¯å³AUC
    auc = 0.
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    print "the auc is %s."%auc
    x = [_v[0] for _v in xy_arr]
    y = [_v[1] for _v in xy_arr]
    pl.title("%s ROC curve (AUC = %.4f)" % (title, auc))
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.plot(x, y)
    pl.savefig(roc_save_path)
    #pl.show()


if __name__ == '__main__':
    file_path = sys.argv[1]
    roc_save_path = sys.argv[2]
    calculate_roc_auc(file_path, roc_save_path, sys.argv[3])

