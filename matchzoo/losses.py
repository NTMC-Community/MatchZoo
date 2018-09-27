"""Self defined loss function."""
import typing
import numpy as np
from keras import backend as K
from keras import layers


__TensorType = typing.Union[np.ndarray, float]
# TODO: Loss function parameters setting.
margin = 1.0
neg_num = 1


def rank_hinge_loss(y_true: __TensorType,
                    y_pred: __TensorType) -> __TensorType:
    """
    Calculate rank hinge loss.

    Support user defined :attr:`margin` and :attr:`neg_num`.

    :param y_true: Label.
    :param y_pred: Predicted result.
    :return: Hinge loss computed by user-defined margin.
    """
    y_pos = layers.Lambda(lambda a: a[::(neg_num+1), :],
                          output_shape=(1,))(y_pred)
    y_neg = []
    for neg_idx in range(neg_num):
        y_neg.append(layers.Lambda(lambda a: a[(neg_idx+1)::(neg_num+1), :],
                                   output_shape=(1,))(y_pred))
    y_neg = K.mean(K.concatenate(y_neg, axis=-1), axis=-1, keepdims=True)
    loss = K.maximum(0., margin + y_neg - y_pos)
    return K.mean(loss)


def rank_crossentropy_loss(y_true: __TensorType,
                           y_pred: __TensorType) -> __TensorType:
    """
    Calculate rank cross entropy loss.

    Support user defined :attr:`neg_num`.

    :param y_true: Label.
    :param y_pred: Predicted result.
    :return: Crossentropy loss computed by user-defined negative number.
    """
    logits = layers.Lambda(lambda a: a[::(neg_num + 1), :])(y_pred)
    labels = layers.Lambda(lambda a: a[::(neg_num + 1), :])(y_true)
    logits, labels = [logits], [labels]
    for neg_idx in range(neg_num):
        neg_logits = layers.Lambda(lambda a:
                                   a[neg_idx+1::(neg_num+1), :])(y_pred)
        neg_labels = layers.Lambda(lambda a:
                                   a[neg_idx+1::(neg_num+1), :])(y_true)
        logits.append(neg_logits)
        labels.append(neg_labels)
    logits = K.concatenate(logits, axis=-1)
    labels = K.concatenate(labels, axis=-1)
    return -K.mean(K.sum(labels*K.log(K.softmax(logits)), axis=-1))
