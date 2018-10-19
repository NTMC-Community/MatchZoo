"""Backend utils."""

from keras import layers
from keras import backend as K


def tensor_mul_tensors(tensor, tensors):
	"""Element wise multiply tensor with list of tensors."""
	tensors = K.stack(tensors)
	def element_wise_multiply(current_tensor):
		"""Do element-wise multiplication."""
		return K.sum(layers.multiply([tensor, current_tensor]),
		             keepdims=True)
	return K.map_fn(element_wise_multiply, tensors)
