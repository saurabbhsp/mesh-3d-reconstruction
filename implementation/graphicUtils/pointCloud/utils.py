import numpy as np


def sample_points(x, n_samples, axis=0, replace=True):
    """This method will sample from point clouds"""
    n_original = x.shape[axis]
    indices = np.random.choice(n_original, n_samples, replace=replace)
    return x.take(indices, axis=axis)
