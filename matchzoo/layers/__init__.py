from .attention_layer import AttentionLayer
from .multi_perspective_layer import MultiPerspectiveLayer
from .matching_layer import MatchingLayer
from .dynamic_pooling_layer import DynamicPoolingLayer

layer_dict = {
    "MatchingLayer": MatchingLayer,
    "DynamicPoolingLayer": DynamicPoolingLayer
}
