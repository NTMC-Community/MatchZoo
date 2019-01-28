from pathlib import Path

USER_DIR = Path.expanduser(Path('~')).joinpath('.matchzoo')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()
USER_TUNED_MODELS_DIR = USER_DIR.joinpath('tuned_models')

from .logger import logger
from .version import __version__

from .data_pack import DataPack
from .data_pack import pack
from .data_pack import load_data_pack
from . import metrics
from . import tasks

from .utils import *

from .preprocessors.units.chain_transform import chain_transform
from . import preprocessors
from . import data_generator

from . import metrics
from . import losses
from . import engine
from . import models
from . import embedding
from . import datasets
from . import layers
from . import contrib

from .engine import hyper_spaces
from .engine.base_model import load_model
from .engine.base_preprocessor import load_preprocessor
from .engine import callbacks

from .embedding.embedding import Embedding

from . import tune
from .prepare import prepare, Preparer
