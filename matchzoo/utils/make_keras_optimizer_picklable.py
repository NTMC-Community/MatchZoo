import keras


def make_keras_optimizer_picklable():
    """
    Fix https://github.com/NTMC-Community/MatchZoo/issues/726.

    This function changes how keras behaves, use with caution.
    """
    def __getstate__(self):
        return keras.optimizers.serialize(self)

    def __setstate__(self, state):
        optimizer = keras.optimizers.deserialize(state)
        self.__dict__ = optimizer.__dict__

    cls = keras.optimizers.Optimizer
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
