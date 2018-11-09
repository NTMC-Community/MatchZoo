"""Matchzoo pair generator."""

import pandas as pd

from . import DataPack


def upsample(data_pack, num_dup=0, num_neg=0):
    pairs = []
    groups = data_pack.relation.sort_values(
        'label', ascending=False).groupby('id_left')
    for idx, group in groups:
        labels = group.label.unique()
        for label in labels:
            pos_samples = group[group.label == label]
            pos_samples = pd.concat([pos_samples] * num_dup)
            neg_samples = group[group.label < label]
            for _, pos_sample in pos_samples.iterrows():
                pos_sample = pd.DataFrame([pos_sample])
                if len(neg_samples) >= num_neg:
                    neg_sample = neg_samples.sample(num_neg,
                                                    replace=False)
                    pairs.extend((pos_sample, neg_sample))
    new_relation = pd.concat(pairs, ignore_index=True)
    return DataPack(relation=new_relation,
                    left=data_pack.left.copy(),
                    right=data_pack.right.copy())
