"""
Debugging utilities
"""

import numpy as np
from numpy.typing import ArrayLike


def rand_sparce_matrix(n: int, m: int, p0: float) -> ArrayLike:
    """
    Generate a random matrix mxn with elements between -1 and 1 and a lot of 0 elements. Lower values of
    `p0` lead to more zeros.
    """
    mat = np.random.rand(n,m)
    neg_idx = np.where(mat < p0)
    zer_idx = np.where(np.logical_and(mat >= p0, mat <= (1-p0)))
    pos_idx = np.where(mat > 1-p0)
    mat[neg_idx] -= p0
    mat[neg_idx] /= p0
    mat[zer_idx] *= 0
    mat[pos_idx] -= 1-p0
    mat[pos_idx] /= 1-p0
    
    return mat
    
def rand_sparce_pos_matrix(n: int, m: int, p0: float) -> ArrayLike:
    """
    Generate a random matrix mxn with elements between 0 and 1 and a lot of 0 elements. Lower values of
    `p0` lead to more zeros.
    """
    mat = np.random.rand(n,m)
    neg_idx = np.where(mat < p0)
    pos_idx = np.where(mat >= p0)
    mat[neg_idx] *= 0
    mat[pos_idx] -= p0
    mat[pos_idx] /= 1-p0
    
    return mat
