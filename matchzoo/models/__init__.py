from .naive_model import NaiveModel
from .dssm_model import DSSMModel
from .cdssm_model import CDSSMModel
from .dense_baseline_model import DenseBaselineModel
from .arci_model import ArcI
from .arcii import ArcII
from .match_pyramid import MatchPyramid
from .knrm_model import KNRM
from .conv_knrm_model import ConvKNRM
from .duet import DUET
from .drmmtks_model import DRMMTKSModel
from .drmm import DRMM
from .anmm_model import ANMMModel
from .mvlstm import MVLSTM

import matchzoo
def list_available():
    return matchzoo.engine.BaseModel.__subclasses__()
