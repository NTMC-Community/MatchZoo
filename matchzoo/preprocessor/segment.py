"""Convert list of input into class:`DataPack` expected format."""

import pandas as pd

from matchzoo import datapack


def segment(inputs: list, stage: str) -> datapack.DataPack:
    """
    Convert user input into :class:`DataPack` consist of two tables.

    :param inputs: Raw user inputs, list of tuples.
    :param stage: `train`, `evaluate`, or `predict`.

    :return: User input into a :class:`DataPack` with left, right and
        relation..
    """
    col_all = ['id_left', 'id_right', 'text_left', 'text_right']
    col_relation = ['id_left', 'id_right']

    if stage in ['train', 'evaluate']:
        col_relation.append('label')
        col_all.append('label')

    # prepare data pack.
    inputs = pd.DataFrame(inputs, columns=col_all)
    inputs.fillna('missing')  # avoid tokenization exception.

    # Segment input into 3 dataframes.
    relation = inputs[col_relation]

    left = inputs[['id_left', 'text_left']].drop_duplicates(['id_left'])
    left.set_index('id_left', inplace=True)

    right = inputs[['id_right', 'text_right']].drop_duplicates(['id_right'])
    right.set_index('id_right', inplace=True)

    return datapack.DataPack(relation=relation,
                             left=left,
                             right=right)
