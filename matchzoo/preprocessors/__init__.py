from .dssm_preprocessor import DSSMPreprocessor
from .naive_preprocessor import NaivePreprocessor
from .basic_preprocessor import BasicPreprocessor
from .cdssm_preprocessor import CDSSMPreprocessor

import matchzoo


def list_available():
    return matchzoo.engine.BasePreprocessor.__subclasses__()
