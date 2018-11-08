"""Convert list of input into class:`DataPack` expected format."""

import pandas as pd

from matchzoo import datapack


def infer_stage(data: list):
    if data and len(data[0]) == 5:
        return 'train'
    elif len(data[0]) == 4:
        return 'predict'
    else:
        raise ValueError('Invalid data format.')


def pack(data: list) -> datapack.DataPack:
    """
    Pack user input into :class:`DataPack`.

    :param data: Raw user inputs, list of tuples.

    :return: User input into a :class:`DataPack` with left, right and
        relation..
    """
    col_all = ['id_left', 'id_right', 'text_left', 'text_right']
    col_relation = ['id_left', 'id_right']

    if infer_stage(data) == 'train':
        col_relation.append('label')
        col_all.append('label')

    # prepare data pack.
    df = pd.DataFrame(data, columns=col_all)
    df.fillna('missing')  # avoid tokenization exception.

    # Segment input into 3 dataframes.
    relation = df[col_relation]

    left = df[['id_left', 'text_left']].drop_duplicates(['id_left'])
    left.set_index('id_left', inplace=True)
    # Infer the length of the text left
    left['length_left'] = left.apply(lambda r: len(r['text_left']), axis=1)

    right = df[['id_right', 'text_right']].drop_duplicates(['id_right'])
    right.set_index('id_right', inplace=True)
    # Infer the length of the text right
    right['length_right'] = right.apply(lambda r: len(r['text_right']), axis=1)

    return datapack.DataPack(relation=relation,
                             left=left,
                             right=right)
