import numpy as np
import numpy.linalg as la
import os

from pathlib import Path

def get_most_recent_file(str):
    path = Path(str)
    try:
        files = [f for f in path.iterdir() if f.is_file()]
        if not files:
            return None
        most_recent_file = max(files, key=os.path.getmtime)
        return most_recent_file
    except FileNotFoundError:
        return None

def channels_first(a: np.ndarray):
    return a.swapaxes(a.ndim - 3, a.ndim - 1).swapaxes(a.ndim - 2, a.ndim - 1)

def channels_last(a: np.ndarray):
    return a.swapaxes(a.ndim - 3, a.ndim - 1).swapaxes(a.ndim - 3, a.ndim - 2)

def svd_reduce(a: np.ndarray, rank):
    svd_tuple = la.svd(a)
    u = svd_tuple[0][..., :, :rank]
    s = svd_tuple[1][..., :rank]
    s_shape = s.shape[:-1]
    s = s.reshape(-1, rank)
    s = np.stack([np.diag(vector) for vector in s])
    s = s.reshape(*s_shape, rank, rank)
    vt = svd_tuple[2][..., :rank, :]
    return u @ s @ vt

def frob_distance_sqr(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape, "arrays should be same shape"
    rows, columns = a.shape[-2:]
    a = a.reshape(-1, rows, columns)
    b = b.reshape(-1, rows, columns)
    return np.sum(np.stack([la.norm(a[i] - b[i], ord='fro') ** 2 for i in range(a.shape[0])]))

