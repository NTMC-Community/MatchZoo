"""Backend utils."""

from keras import layers
from keras import backend as K


def tensor_mul_tensors(tensor, tensors):
    """Element wise multiply tensor with list of tensors."""
    tensors = K.stack(tensors)

    def element_wise_multiply(current_tensor):
        """Do element-wise multiplication."""
        return layers.multiply([tensor, current_tensor])
    return K.map_fn(element_wise_multiply, tensors)


def tensor_mul_tensors_with_max_pooling(tensor, tensors):
    """Multiply tensor with list of tensors and retain the maximum value."""
    return K.max(tensor_mul_tensors(tensor, tensors),
                 axis=0,
                 keepdims=True)


def tensor_dot_tensors(tensor, tensors):
    """Compute cosine similairty between tensor and list of tensors."""
    tensors = K.stack(tensors)

    def cosine_similarity(current_tensor):
        """Compute cosine similarity between tensor and tensor."""
        return K.sum(tensor * current_tensor,
                     axis=-1,
                     keepdims=True)
    return K.map_fn(cosine_similarity, tensors)


def tensors_dot_tensors(tensors_lt, tensors_rt):
    """Compute cosine similairty between two list of tensors."""
    tensors_lt = K.stack(tensors_lt)

    def cosine_similarity(current_tensor):
        """Cosine similarity between tensor and list of tensors."""
        return tensor_dot_tensors(current_tensor, tensors_rt)
    return K.map_fn(cosine_similarity, tensors_lt)
