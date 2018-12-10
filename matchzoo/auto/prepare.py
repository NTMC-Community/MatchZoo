"""Prepare mode, preprocessor, and data pack."""

import copy
import typing

import numpy as np

import matchzoo
from matchzoo import tasks
from matchzoo import models


def prepare(
    model,
    data_pack,
    preprocessor=None,
    verbose=1
) -> typing.Tuple[matchzoo.engine.BaseModel,
                  matchzoo.DataPack,
                  matchzoo.engine.BasePreprocessor]:
    """
    Prepare mode, preprocessor, and data pack.

    :param model:
    :param data_pack:
    :param preprocessor:
    :param verbose:
    :return:

    """
    params = copy.deepcopy(model.params)

    if preprocessor:
        new_preprocessor = copy.deepcopy(preprocessor)
    else:
        new_preprocessor = model.get_default_preprocessor()

    train_pack_processed = new_preprocessor.fit_transform(data_pack, verbose)

    if not params['task']:
        params['task'] = _guess_task(data_pack)

    context = {}
    if 'input_shapes' in new_preprocessor.context:
        context['input_shapes'] = new_preprocessor.context['input_shapes']

    if isinstance(model, models.DSSMModel):
        params['input_shapes'] = context['input_shapes']

    if 'with_embedding' in params:
        term_index = new_preprocessor.context['vocab_unit'].state['term_index']
        vocab_size = len(term_index) + 1
        params['embedding_input_dim'] = vocab_size

    new_model = type(model)(params=params)
    new_model.guess_and_fill_missing_params(verbose=verbose)
    new_model.build()
    new_model.compile()

    return new_model, train_pack_processed, new_preprocessor


def _guess_task(train_pack):
    if np.issubdtype(train_pack.relation['label'].dtype, np.number):
        return tasks.Ranking()
    elif np.issubdtype(train_pack.relation['label'].dtype, list):
        num_classes = int(train_pack.relation['label'].apply(len).max())
        return tasks.Classification(num_classes)
