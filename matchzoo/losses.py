"""Self define loss function."""
from keras import backend as K
from keras import layers

# TODO: Loss function parameters setting.

_margin = 1.
_neg_num = 1


def rank_hinge_loss(y_true, y_pred):
    """
    Margin and sample-number based ranking loss.

    :param y_true: Label.
    :param y_pred: Predict result.
    :return: Hinge loss computed by user-defined margin.
    """
    y_pos = layers.Lambda(lambda a: a[::(_neg_num+1), :],
                          output_shape=(1,))(y_pred)
    y_neg = []
    for neg_idx in range(_neg_num):
        y_neg.append(layers.Lambda(lambda a: a[(neg_idx+1)::(_neg_num+1), :],
                                   output_shape=(1,))(y_pred))
    y_neg = K.max(K.concatenate(y_neg, axis=-1), axis=-1, keepdims=True)
    loss = K.maximum(0., _margin + y_neg - y_pos)
    return K.mean(loss)


def rank_crossentropy_loss(y_true, y_pred):
    """
    Sample-number based crossentropy loss.

    :param y_true: Label.
    :param y_pred: Predict result.
    :return: Crossentropy loss computed by user-defined negative number.
    """
    pos_logits = layers.Lambda(lambda a: a[::(_neg_num+1), :])(y_pred)
    pos_labels = layers.Lambda(lambda a: a[::(_neg_num+1), :])(y_true)
    logits_list, labels_list = [pos_logits], [pos_labels]
    for neg_idx in range(_neg_num):
        neg_logits = layers.Lambda(lambda a:
                                   a[neg_idx+1::(_neg_num+1), :])(y_pred)
        neg_labels = layers.Lambda(lambda a:
                                   a[neg_idx+1::(_neg_num+1), :])(y_true)
        logits_list.append(neg_logits)
        labels_list.append(neg_labels)
    logits = K.concatenate(logits_list, axis=-1)
    labels = K.concatenate(labels_list, axis=-1)
    return -K.mean(K.sum(labels*K.log(K.softmax(logits)), axis=-1))
