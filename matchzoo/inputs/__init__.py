# note 
from .rank_data_generator import PairGenerator
from .rank_data_generator import ListGenerator
from .drmm_data_generator import DRMM_PairGenerator
from .drmm_data_generator import DRMM_ListGenerator
#from ..utils import rank_io
import six
from keras.utils.generic_utils import deserialize_keras_object

def serialize(generator):
    return generator.__name__


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)

