"""The rank hinge loss."""

import numpy as np
import tensorflow as tf
from keras import layers, backend as K
from keras.losses import Loss
from keras.utils import losses_utils


class RankHingeLoss(Loss):
    """
    Rank hinge loss.

    Examples:
        >>> from keras import backend as K
        >>> x_pred = K.variable(np.array([[1.0], [1.2], [0.8], [1.4]]))
        >>> x_true = K.variable(np.array([[1], [0], [1], [0]]))
        >>> expect = ((1.0 + 1.2 - 1.0) + (1.0 + 1.4 - 0.8)) / 2
        >>> expect
        1.4
        >>> loss = K.eval(RankHingeLoss(num_neg=1, margin=1.0)(x_true, x_pred))
        >>> np.isclose(loss, expect)
        True

    """

    def __init__(self, num_neg: int = 1, margin: float = 1.0):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: number of negative instances in hinge loss.
        :param margin: the margin between positive and negative scores.
        """
        super().__init__(reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                         name="rank_hinge")

        self._num_neg = num_neg
        self._margin = margin

    def call(self, y_true: np.array, y_pred: np.array,
             sample_weight=None) -> np.array:
        """
        Calculate rank hinge loss.

        :param y_true: Label.
        :param y_pred: Predicted result.
        :return: Hinge loss computed by user-defined margin.
        """
        y_pos = layers.Lambda(lambda a: a[::(self._num_neg + 1), :],
                              output_shape=(1,))(y_pred)
        y_neg = []
        for neg_idx in range(self._num_neg):
            y_neg.append(
                layers.Lambda(
                    lambda a: a[(neg_idx + 1)::(self._num_neg + 1), :],
                    output_shape=(1,))(y_pred))
        y_neg = tf.concat(y_neg, axis=-1)
        y_neg = tf.reduce_mean(y_neg, axis=-1, keepdims=True)
        loss = tf.maximum(0., self._margin + y_neg - y_pos)
        return losses_utils.compute_weighted_loss(
            loss, sample_weight, reduction=self.reduction)

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @property
    def margin(self):
        """`margin` getter."""
        return self._margin
