from .precision import Precision
from .average_precision import AveragePrecision
from .discounted_cumulative_gain import DiscountedCumulativeGain
from .mean_reciprocal_rank import MeanReciprocalRank
from .mean_average_precision import MeanAveragePrecision
from .normalized_discounted_cumulative_gain import \
    NormalizedDiscountedCumulativeGain


def list_available() -> list:
    from matchzoo.engine.base_metric import BaseMetric
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseMetric)
