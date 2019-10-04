"""The rank cross entropy loss."""

import numpy as np
import tensorflow as tf
from keras import layers, backend as K
from keras.losses import Loss
from keras.utils import losses_utils


class RankCrossEntropyLoss(Loss):
    """
    Rank cross entropy loss.

    Examples:
        >>> from keras import backend as K
        >>> softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=0)
        >>> x_pred = K.variable(np.array([[1.0], [1.2], [0.8]]))
        >>> x_true = K.variable(np.array([[1], [0], [0]]))
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
        super().__init__(reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                         name="rank_crossentropy")
        self._num_neg = num_neg

    def call(self, y_true: np.array, y_pred: np.array,
             sample_weight=None) -> np.array:
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
        logits = tf.concat(logits, axis=-1)
        labels = tf.concat(labels, axis=-1)
        smoothed_prob = tf.nn.softmax(logits) + np.finfo(float).eps
        loss = -(tf.reduce_sum(labels * tf.math.log(smoothed_prob), axis=-1))
        return losses_utils.compute_weighted_loss(
            loss, sample_weight, reduction=self.reduction)

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg
