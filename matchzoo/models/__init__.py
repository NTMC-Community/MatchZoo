from .naive import Naive
from .dssm import DSSM
from .cdssm import CDSSM
from .dense_baseline import DenseBaseline
from .arci import ArcI
from .arcii import ArcII
from .match_pyramid import MatchPyramid
from .knrm import KNRM
from .conv_knrm import ConvKNRM
from .duet import DUET
from .drmmtks import DRMMTKS
from .drmm import DRMM
from .anmm import ANMM
from .mvlstm import MVLSTM


def list_available() -> list:
    from matchzoo.engine.base_model import BaseModel
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseModel)
