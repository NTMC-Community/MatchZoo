"""Self define loss function."""
from keras import backend as K
from keras.layers import Lambda

_margin = 1.
_neg_num = 1


def rank_hinge_loss(y_true, y_pred):
    """
    Support user-defined margin value.

    :param y_true: Label.
    :param y_pred: Predict result.
    :return: Hinge loss computed by user-defined margin.
    """
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg_list = []
    for i in range(_neg_num):
        y_neg_list.append(Lambda(lambda a: a[1::2, :],
                                 output_shape=(1,))(y_pred))
    y_neg = K.concatenate(y_neg_list, axis=-1)
    loss = K.maximum(0., _margin + y_neg - y_pos)
    return K.mean(loss)


def rank_crossentropy_loss(y_true, y_pred):
    """
    Support user-defined negative sample number.

    :param y_true: Label.
    :param y_pred: Predict result.
    :return: Crossentropy loss computed by user-defined negative number.
    """
    y_pos_logits = Lambda(lambda a: a[::(_neg_num+1), :])(y_pred)
    y_pos_labels = Lambda(lambda a: a[::(_neg_num+1), :])(y_true)
    logits_list, labels_list = [y_pos_logits], [y_pos_labels]
    for i in range(_neg_num):
        y_neg_logits = Lambda(lambda a: a[i+1::(_neg_num+1), :])(y_pred)
        y_neg_labels = Lambda(lambda a: a[i+1::(_neg_num+1), :])(y_true)
        logits_list.append(y_neg_logits)
        labels_list.append(y_neg_labels)
    logits = K.concatenate(logits_list, axis=-1)
    labels = K.concatenate(labels_list, axis=-1)
    return -K.mean(K.sum(labels*K.log(K.softmax(logits)), axis=-1))
