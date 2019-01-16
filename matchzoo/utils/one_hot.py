"""One hot vectors."""
import numpy as np


def one_hot(indices: int, num_classes: int) -> np.ndarray:
    """:return: A one-hot encoded vector."""
    vec = np.zeros((num_classes,), dtype=np.int64)
    vec[indices] = 1
    return vec
