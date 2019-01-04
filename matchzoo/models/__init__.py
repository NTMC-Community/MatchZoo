from .naive_model import NaiveModel
from .dssm_model import DSSMModel
from .cdssm_model import CDSSMModel
from .dense_baseline_model import DenseBaselineModel
from .arci_model import ArcIModel
from .arcii_model import ArcIIModel
from .match_pyramid import MatchPyramid
from .knrm_model import KNRMModel
from .conv_knrm_model import ConvKNRMModel
from .duet import DUET
from .drmmtks_model import DRMMTKSModel
from .drmm import DRMM
from .anmm_model import ANMMModel
from .match_lstm import MatchLSTM
from .mvlstm import MVLSTM
from .bimpm_model import BimpmModel

import matchzoo
def list_available():
    return matchzoo.engine.BaseModel.__subclasses__()
