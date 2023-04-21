"""utils functions for sparse regression codes"""
import random
import numpy as np


def clip(z, epsilon=1e-6):
    if z < -epsilon:
        return -epsilon
    elif z > epsilon:
        return epsilon
    else:
        return z


def sparse_vector_generator(B, L, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_sparse = [0] * B * L
    l_shuffle = random.shuffle
