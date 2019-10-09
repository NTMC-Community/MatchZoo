from pathlib import Path

USER_DIR = Path.expanduser(Path('~')).joinpath('.matchzoo')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()
USER_TUNED_MODELS_DIR = USER_DIR.joinpath('tuned_models')

from .version import __version__

from .data_pack import DataPack
from .data_pack import pack
from .data_pack import load_data_pack

from . import metrics
from . import tasks

from . import preprocessors
from . import data_generator
from .data_generator import DataGenerator
from .data_generator import DataGeneratorBuilder

from .preprocessors.chain_transform import chain_transform

from . import metrics
from . import losses
from . import engine
from . import models
from . import embedding
from . import datasets
from . import layers
from . import auto
from . import contrib

from .engine import hyper_spaces
from .engine.base_model import load_model
from .engine.base_preprocessor import load_preprocessor
from .engine import callbacks
from .engine.param import Param
from .engine.param_table import ParamTable

from .embedding.embedding import Embedding

from .utils import one_hot, make_keras_optimizer_picklable
from .preprocessors.build_unit_from_data_pack import build_unit_from_data_pack
from .preprocessors.build_vocab_unit import build_vocab_unit

# deprecated, should be removed in v2.2
from .contrib.legacy_data_generator import DPoolDataGenerator
from .contrib.legacy_data_generator import DPoolPairDataGenerator
from .contrib.legacy_data_generator import HistogramDataGenerator
from .contrib.legacy_data_generator import HistogramPairDataGenerator
from .contrib.legacy_data_generator import DynamicDataGenerator
from .contrib.legacy_data_generator import PairDataGenerator
