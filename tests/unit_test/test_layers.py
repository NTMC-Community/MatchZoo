import numpy as np
import pytest
from keras import backend as K

from matchzoo import layers


def test_matching_layers():
    s1_value = np.array([[[1, 2], [2, 3], [3, 4]],
                         [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
                        ])
    s2_value = np.array([[[1, 2], [2, 3]],
                         [[0.1, 0.2], [0.2, 0.3]]
                        ])
    s3_value = np.array([[[1, 2], [2, 3]],
                         [[0.1, 0.2], [0.2, 0.3]],
                         [[0.1, 0.2], [0.2, 0.3]]
                        ])
    s1_tensor = K.variable(s1_value)
    s2_tensor = K.variable(s2_value)
    s3_tensor = K.variable(s3_value)
    for matching_type in ['dot', 'mul', 'plus', 'minus', 'concat']:
        model = layers.MatchingLayer(matching_type=matching_type)([s1_tensor, s2_tensor])
        ret = K.eval(model)
    with pytest.raises(ValueError):
        layers.MatchingLayer(matching_type='error')
    with pytest.raises(ValueError):
        layers.MatchingLayer()([s1_tensor, s3_tensor])
