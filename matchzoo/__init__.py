from pathlib import Path

USER_DIR = Path.expanduser(Path('~')).joinpath('.matchzoo')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()

from .logger import logger
from .version import __version__

from . import processor_units
from .processor_units import chain_transform, ProcessorUnit

from .data_pack import DataPack, pack, build_vocab_unit, \
    build_unit_from_data_pack, load_data_pack

from .data_generator import DataGenerator
from .data_generator import PairDataGenerator
from .data_generator import DynamicDataGenerator

from . import tasks
from . import metrics
from . import losses
from . import engine
from . import preprocessors
from . import models
from . import datasets
from . import embedding

from .engine import load_model, load_preprocessor
from .auto import Director
