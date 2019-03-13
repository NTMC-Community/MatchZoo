"""The rank cross entropy loss."""
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


class RankCrossEntropyLoss(object):
    """
    Rank cross entropy loss.

    Examples:
        >>> import tensorflow as tf
        >>> import tensorflow.keras.backend as K
        >>> softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=0)
        >>> x_pred = tf.variable(np.array([[1.0], [1.2], [0.8]]))
        >>> x_true = tf.variable(np.array([[1], [0], [0]]))
        >>> expect = -np.log(softmax(np.array([[1.0], [1.2], [0.8]])))
        >>> loss = K.eval(RankCrossEntropyLoss(num_neg=2)(x_true, x_pred))
        >>> np.isclose(loss, expect[0]).all()
        True

    """

    def __init__(self, num_neg: int = 1):
        """
        :class:`RankCrossEntropyLoss` constructor.

        :param num_neg: number of negative instances in cross entropy loss.
        """
        self._num_neg = num_neg

    def __call__(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Calculate rank cross entropy loss.

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
        logits = tf.keras.backend.concatenate(logits, axis=-1)
        labels = tf.keras.backend.concatenate(labels, axis=-1)
        return -tf.keras.backend.mean(tf.keras.backend.sum(
            labels * tf.keras.backend.log(tf.keras.backend.softmax(logits)),
            axis=-1))

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg
