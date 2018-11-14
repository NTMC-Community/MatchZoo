import numpy as np

from keras import layers, backend as K


class RankCrossEntropyLoss(object):
    def __init__(self, num_neg):
        self._num_neg = num_neg

    def __call__(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Calculate rank cross entropy loss.

        Support user defined :attr:`neg_num`.

        :param y_true: Label.
        :param y_pred: Predicted result.
        :return: Crossentropy loss computed by user-defined negative number.
        """
        logits = layers.Lambda(lambda a: a[::(self._num_neg + 1), :])(y_pred)
        labels = layers.Lambda(lambda a: a[::(self._num_neg + 1), :])(y_true)
        logits, labels = [logits], [labels]
        for neg_idx in range(self._num_neg):
            neg_logits = layers.Lambda(
                lambda a: a[neg_idx + 1::(self._num_neg + 1), :])(y_pred)
            neg_labels = layers.Lambda(
                lambda a: a[neg_idx + 1::(self._num_neg + 1), :])(y_true)
            logits.append(neg_logits)
            labels.append(neg_labels)
        logits = K.concatenate(logits, axis=-1)
        labels = K.concatenate(labels, axis=-1)
        return -K.mean(K.sum(labels * K.log(K.softmax(logits)), axis=-1))
