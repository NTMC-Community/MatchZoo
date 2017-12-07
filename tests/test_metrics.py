
import os
import sys

sys.path.append('../matchzoo/metrics/')
import rank_evaluations


if __name__=='__main__':
    trec_file = './ranklist'
    qrel_file = './qrels'

    rank_eval = rank_evaluations.rank_eval(rel_threshold = 0.)

    scores = {}
    for line in open(trec_file, 'r'):
        r = line.strip().split()
        qid = r[0]
        did = r[2]
        score = float(r[4])
        if qid not in scores:
            scores[qid] = {}
        scores[qid][did] = score
    qrels = {}
    for line in open(qrel_file, 'r'):
        r = line.strip().split()
        if r[0] not in qrels:
            qrels[r[0]] = {}
        qrels[r[0]][r[2]] = int(r[3])


    metrics=['map', 'ndcg@5', 'ndcg@10', 'p@5', 'p@10']
    #print rank_eval.eval(y_pred = [9, 7, 6, 5, 1], y_true = [0, 1, 0, 0, 2], metrics=metrics)
    res = dict([[k, 0.] for k in metrics])
    map, ndcg_5, ndcg_10, p_5, p_10 = 0., 0., 0., 0., 0.
    num_valid = 0
    for qid, dinfo in scores.items():
        y_pred = []
        y_true = []
        for did, s in dinfo.items():
            y_pred.append(s)
            y_true.append(qrels[qid][did])
        map += rank_evaluations.eval_map(y_true = y_true, y_pred = y_pred)
        ndcg_5 += rank_evaluations.eval_ndcg(y_true = y_true, y_pred = y_pred, k = 5)
        ndcg_10 += rank_evaluations.eval_ndcg(y_true = y_true, y_pred = y_pred, k = 10)
        p_5 += rank_evaluations.eval_precision(y_true = y_true, y_pred = y_pred, k = 5)
        p_10 += rank_evaluations.eval_precision(y_true = y_true, y_pred = y_pred, k = 10)
        curr_res = rank_eval.eval(y_true = y_true, y_pred = y_pred,
                metrics = metrics)
        for k,v in curr_res.items():
            res[k] += v
        num_valid += 1
    print '  '.join(['%s:%f'%(k,v/num_valid) for k, v in res.items()]), ' ...'

    print 'map: ',  map/num_valid
    print 'ndcg@5: ',  ndcg_5/num_valid
    print 'ndcg@10: ',  ndcg_10/num_valid
    print 'p@5: ',  p_5/num_valid
    print 'p@10: ',  p_10/num_valid
