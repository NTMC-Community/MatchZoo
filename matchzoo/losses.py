"""Self define loss function."""
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape
from keras.layers import Permute

_margin = 1.
_neg_num = 1


class SliceTensor(Layer):
    """
    Matchzoo :class:`SliceTensor` slice keras tensor.

    Example:
        >>> input_tensor = K.variable([[
        ...     [0, 0, 0],
        ...     [1, 1, 1],
        ...     [2, 2, 2],
        ...     [3, 3, 3]
        ... ]])
        >>> K.int_shape(input_tensor)
        (1, 4, 3)
        >>> output_tensor = SliceTensor(0,2,0)(input_tensor)
        >>> K.int_shape(output_tensor)
        (1, 2, 3)
        >>> K.eval(output_tensor)
        array([[[0., 0., 0.],
                [2., 2., 2.]]], dtype=float32)
    """

    def __init__(self, axis, slices, index, mode="tensor", **kwargs):
        """
        Initialize :class:`SliceTensor`.

        :param axis: Slice axis (ignore batch_size dimension).
        :param slices: Number of divisions.
        :param index: Start index.
        :param mode: Slice mode, for "tensor" of "loss"
        :param kwargs: kwargs.
        """
        self.slices = int(slices)
        self.index = int(index)
        self.axis = int(axis)
        self.mode = mode
        super(SliceTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Keras required build function.

        :param input_shape: Shape of input tensor.
        :return: Standard build return.
        """
        super(SliceTensor, self).build(input_shape)

    def call(self, x):
        """
        Keras required call function.

        :param x: Input tensor.
        :return: Sliced tensor.
        """
        input_shape = K.int_shape(x)
        if self.mode == "tensor":
            input_shape = list(input_shape[1:])
        else:
            input_shape = list(input_shape)
            x = K.expand_dims(x, 0)
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
            output_shape[0], output_shape[self.axis] = \
                output_shape[self.axis], output_shape[0]
        output = Reshape(tuple(output_shape))(output)
        output = Permute(tuple(permute_shape))(output)
        return output

    def compute_output_shape(self, input_shape):
        """
        Compute output shape.

        :param input_shape: Shape of input tensor.
        :return: Shape of output tensor.
        """
        if isinstance(input_shape, tuple):
            input_shape = list(input_shape)
        if self.axis >= 0:
            self.axis += 1
        input_shape[self.axis] = int(input_shape[self.axis]/self.slices)
        return tuple(input_shape)


def rank_hinge_loss(y_true, y_pred):
    """
    Support user-defined margin value.

    :param y_true: Label.
    :param y_pred: Predict result.
    :return: Hinge loss computed by user-defined margin.
    """
    y_pos = SliceTensor(0, _neg_num+1, 0, mode='loss')(y_pred)
    y_neg_list = []
    for i in range(_neg_num):
        y_neg_list.append(SliceTensor(0, _neg_num+1, i+1, mode='loss')(y_pred))
    y_neg = K.concatenate(y_neg_list, axis=-1)
    print(K.eval(y_neg))
    loss = K.maximum(0., _margin + y_neg - y_pos)
    return K.mean(loss)


def rank_crossentropy_loss(y_true, y_pred):
    """
    Support user-defined negative sample number.

    :param y_true: Label.
    :param y_pred: Predict result.
    :return: Crossentropy loss computed by user-defined negative number.
    """
    y_pos_logits = SliceTensor(0, _neg_num+1, 0, mode='loss')(y_pred)
    y_pos_labels = SliceTensor(0, _neg_num+1, 0, mode='loss')(y_true)
    logits_list, labels_list = [y_pos_logits], [y_pos_labels]
    for i in range(_neg_num):
        y_neg_logits = SliceTensor(0, _neg_num+1, i+1, mode='loss')(y_pred)
        y_neg_labels = SliceTensor(0, _neg_num+1, i+1, mode='loss')(y_true)
        logits_list.append(y_neg_logits)
        labels_list.append(y_neg_labels)
    logits = K.concatenate(logits_list, axis=-1)
    labels = K.concatenate(labels_list, axis=-1)
    return -K.mean(K.sum(labels*K.log(K.softmax(logits)), axis=-1))
