from .naive_model import NaiveModel
from .dssm_model import DSSMModel
from .cdssm_model import CDSSMModel
from .dense_baseline_model import DenseBaselineModel
from .arci_model import ArcIModel
from .knrm_model import KNRMModel
from .conv_knrm_model import ConvKNRMModel
from .duet_model import DUETModel
from .drmmtks_model import DRMMTKSModel
from .bimpm_model import BimpmModel

import matchzoo
def list_available():
    return matchzoo.engine.BaseModel.__subclasses__()