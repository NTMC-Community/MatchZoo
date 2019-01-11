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

import matchzoo
def list_available():
    return matchzoo.engine.BaseModel.__subclasses__()
