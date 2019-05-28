from .matching_layer import MatchingLayer
from .dynamic_pooling_layer import DynamicPoolingLayer
from .layer_normalization import LayerNormalization
from .transformer import Transformer

layer_dict = {
    "MatchingLayer": MatchingLayer,
    "DynamicPoolingLayer": DynamicPoolingLayer,
    "LayerNormalization": LayerNormalization,
    "Transformer": Transformer
}
