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

    This handles interaction among data, model, and preprocessor
    automatically. For example, some model like `DSSM` have dynamic input
    shapes based on the result of word hashing. Some models have an
    embedding layer which dimension is related to the data's vocabulary
    size. `prepare` takes care of all that and returns properly prepared
    model, data, and preprocessor for you.

    :param model:
    :param data_pack:
    :param preprocessor: If not set, use the model's default preprocessor.
    :param verbose: Verbosity, 0 or 1.
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

    if 'input_shapes' in new_preprocessor.context:
        params['input_shapes'] = new_preprocessor.context['input_shapes']

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
