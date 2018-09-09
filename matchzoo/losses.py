"""self define loss function.
"""
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape
from keras.layers import Permute

_margin = 1.
_neg_num = 1


class SliceTensor(Layer):

    def __init__(self, axis, slices, index, **kwargs):
        self.slices = int(slices)
        self.index = int(index)
        self.axis = int(axis)
        super(SliceTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SliceTensor, self).build(input_shape)

    def call(self, x):
        input_shape = K.int_shape(x)
        if input_shape[0] is None:
            input_shape = list(input_shape[1:])
        else:
            input_shape = list(input_shape)
            x = K.expand_dims(x, axis=0)
        if self.axis < 0:
            self.axis += len(input_shape)
        permute_shape = [i+1 for i in range(len(input_shape))]
        if self.axis != 0:
            permute_shape[self.axis] = 1
            permute_shape[0] = self.axis+1
            x = Permute(tuple(permute_shape))(x)
        reshape_shape = (input_shape[self.axis], -1)
        output = Reshape(reshape_shape)(x)
        output = output[:, self.index::self.slices]

        output_shape = input_shape
        output_shape[self.axis] = int(output_shape[self.axis]/self.slices)
        if self.axis != 0:
            output_shape[0], output_shape[self.axis] = output_shape[self.axis], output_shape[0]
        output = Reshape(tuple(output_shape))(output)
        output = Permute(tuple(permute_shape))(output)
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        if input_shape[0] is None and self.axis >= 0:
            self.axis += 1
        input_shape[self.axis] = input_shape[self.axis]/self.slices
        return tuple(input_shape)


def set_margin(margin=1.0):
    global _margin
    _margin = float(margin)


def set_neg_num(neg_num=1):
    global _neg_num
    _neg_num = neg_num


def rank_hinge_loss(y_true, y_pred):
    y_pos = SliceTensor(0, 2, 0)(y_pred)
    y_neg = SliceTensor(0, 2, 1)(y_pred)
    loss = K.maximum(0., _margin + y_neg - y_pos)
    return K.mean(loss)


def rank_crossentropy_loss(y_true, y_pred):
    y_pos_logits = SliceTensor(0, _neg_num+1, 0)(y_pred)
    y_pos_labels = SliceTensor(0, _neg_num+1, 0)(y_true)
    logits_list, labels_list = [y_pos_logits], [y_pos_labels]
    for i in range(_neg_num):
        y_neg_logits = SliceTensor(0, _neg_num+1, i+1)(y_pred)
        y_neg_labels = SliceTensor(0, _neg_num+1, i+1)(y_true)
        logits_list.append(y_neg_logits)
        labels_list.append(y_neg_labels)
    logits = K.concatenate(logits_list, axis=1)
    labels = K.concatenate(labels_list, axis=1)
    return -K.mean(K.sum(labels*K.log(K.softmax(logits)), axis=0))
