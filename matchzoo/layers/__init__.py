from .matching_layer import MatchingLayer
from .dynamic_pooling_layer import DynamicPoolingLayer
from .matching_tensor_layer import MatchingTensorLayer
from .spatial_gru import SpatialGRU

layer_dict = {
    "MatchingLayer": MatchingLayer,
    "DynamicPoolingLayer": DynamicPoolingLayer
}
