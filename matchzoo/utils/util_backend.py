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

def tensor_mul_tensors_reduce_dim(tensor, tensors):
	"""Element wise multiply tensor with list of tensors, reduce the dimensionality."""
	tensors = K.stack(tensors)
	def element_wise_multiply(current_tensor):
		"""Do element-wise multiplication."""
		return K.sum(layers.multiply([tensor, current_tensor]),
		             keepdims=True)
	return K.map_fn(element_wise_multiply, tensors)

def tensor_mul_tensors_with_max_pooling(tensor, tensors):
	"""Multiply tensor with list of tensors and retain the maxmum value."""
	return K.max(tensor_mul_tensors(tensor, tensors),
	             axis=-1,
	             keepdims=True)
