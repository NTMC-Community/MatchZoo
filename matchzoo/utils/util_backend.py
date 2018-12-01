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
                     axis=0,
                     keepdims=True)
    return K.map_fn(cosine_similarity, tensors)


def tensors_dot_tensors(tensors_lt, tensors_rt):
    """Compute cosine similairty between two list of tensors."""
    tensors_lt = K.stack(tensors_lt)

    def cosine_similarity(current_tensor):
        """Cosine similarity between tensor and list of tensors."""
        return tensor_dot_tensors(current_tensor, tensors_rt)
    return K.map_fn(cosine_similarity, tensors_lt)


#================================================================================
# without mask version
# Attention (tensor v.s. tensors)
#================================================================================

def dot_product_attention(tensor_left, tensors_right, mask_right=None):
    """
    Attention dot_product
    tensor_left: [batch_size, hidden_size]
    tensors_right: [batch_size, sequence_length, hidden_size]
    mask_right: [batch_size, sequence_length]
    :return: [batch_size, hidden_size]
    """
    tensor_left = K.expand_dims(tensor_left, 1)
    passage_prob = softmask(K.sum(tensor_left * tensors_right, axis=2), mask_right)
    passage_rep = K.sum(tensors_right * K.expand_dims(passage_prob, axis=-1), axis=1)
    return passage_rep


def bilinear_attention(question_rep, passage_repres, passage_mask):
    """
    Attention bilinear
    Parameters: similat to dot product attention
    """
    hidden_size = question_rep.get_shape()[1]
    # W_bilinear = tf.get_variable("W_bilinear", shape=[hidden_size, hidden_size], dtype=tf.float32)

    # question_rep = tf.matmul(question_rep, W_bilinear)
    # question_rep = tf.expand_dims(question_rep, 1)
    # passage_prob = softmask(tf.reduce_sum(question_rep * passage_repres, axis=2), passage_mask)
    # passage_rep = tf.reduce_sum(passage_repres * tf.expand_dims(passage_prob, axis=-1), axis=1)
    # return passage_rep
    pass



#================================================================================
# without mask version
# Attention (tensors v.s. tensors)
#================================================================================

def dot_attention(self, tensor_left, tensor_right):
    """
    Compute the attention between elements of two sentences with the dot
    product.
    Params:
        tensor_left: [batch, time_steps, d]
    Returns:
        [batch, time_steps(), times_steps()]
    """
    attn_weights = K.batch_dot(x=tensor_left,
                               y=K.permute_dimensions(tensor_right,
                                                      pattern=(0, 2, 1)))
    return K.permute_dimensions(attn_weights, (0, 2, 1))


def fc_attention(self, tensor_left, tensor_right):
    """
    Compute the attention between elements of two sentences with the fully-connected network.
    """
    tensor_left = K.expand_dims(tensor_left, axis=2)
    tensor_right = K.expand_dims(tensor_right, axis=1)
    tensor_left = K.repeat_elements(tensor_left, tensor_right.shape[2], 2)
    tensor_right = K.repeat_elements(tensor_right, tensor_left.shape[1], 1)
    tensor_merged = K.concatenate([tensor_left, tensor_right], axis=-1)
    middle_output = Dense(128, activation='tanh')(tensor_merged)
    attn_weights = Dense(128)(tensor_merged)
    attn_weights = K.squeeze(attn_weights)

    return attn_weights


#================================================================================
# without mask version
# Functional
#================================================================================


def softmask(input_prob, input_mask=None, eps=1e-6):
    """
    normarlize the probability
    :param input_prob: [batch_size, sequence_length]
    :param input_mask: [batch_size, sequence_length]
    :return: [batch_size, sequence]
    """
    input_prob = K.exp(input_prob)
    if input_mask is not None:
        input_prob = input_prob * input_mask
    input_sum = K.sum(input_prob, axis=1, keepdims=True)
    input_prob = input_prob / (input_sum + eps)
    return input_prob


def soft_alignment(self, attn_weights, tensor_to_align):
    """
    Compute the soft alignment.
    """
    # Subtract the max. from the attention weights to avoid overflows.
    exp = K.exp(attn_weights - K.max(attn_weights, axis=-1, keepdims=True))
    exp_sum = K.sum(exp, axis=-1, keepdims=True)
    softmax = exp / exp_sum

    return K.batch_dot(softmax, tensor_to_align)



