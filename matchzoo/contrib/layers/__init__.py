from .attention_layer import AttentionLayer
from .multi_perspective_layer import MultiPerspectiveLayer
from .matching_tensor_layer import MatchingTensorLayer
from .spatial_gru import SpatialGRU

layer_dict = {
    "MatchingTensorLayer": MatchingTensorLayer,
    "SpatialGRU": SpatialGRU
}
