from .attention_layer import AttentionLayer
from .multi_perspective_layer import MultiPerspectiveLayer
from .matching_tensor_layer import MatchingTensorLayer
from .spatial_gru import SpatialGRU
from .decaying_dropout_layer import DecayingDropoutLayer
from .semantic_composite_layer import EncodingLayer

layer_dict = {
    "MatchingTensorLayer": MatchingTensorLayer,
    "SpatialGRU": SpatialGRU,
    "DecayingDropoutLayer": DecayingDropoutLayer,
    "EncodingLayer": EncodingLayer
}
