"""
Various utilities
"""


from numpy.typing import ArrayLike
import numpy as np

from global_consts import FLOAT_ATOL, FLOAT_RTOL


def check_parallel(a: ArrayLike, b:ArrayLike) -> ArrayLike:
    """
    Given broadcastable shapes a and b of vectors, return a shape containing which vectors in a are
    parallel to which in b. Parallelity is checked by seeing if a.b == |a|.|b|
    """
    return np.isclose( np.einsum("...i,...i->...", a, b), 
                        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis = -1),
                                    rtol = FLOAT_RTOL, atol = FLOAT_ATOL )
