"""
Various utilities
"""


from numpy.typing import ArrayLike
import numpy as np

from global_consts import FLOAT_ATOL, FLOAT_RTOL


def check_parallel(a: ArrayLike, b:ArrayLike, anti_are_parallel = True) -> ArrayLike:
    """
    Given broadcastable shapes a and b of vectors, return a shape containing which vectors in a are
    parallel to which in b. Parallelity is checked by seeing if abs(a.b) == |a|.|b|. If `anti` is
    set to False, andiparallel vectors are not considered parallel, in which case, a.b == |a|.|b| is
    used.
    """
    if anti_are_parallel:
        return np.isclose( np.absolute( np.einsum("...i,...i->...", a, b)), 
                            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis = -1),
                                    rtol = FLOAT_RTOL, atol = FLOAT_ATOL )
    else:
        return np.isclose( np.einsum("...i,...i->...", a, b), 
                            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis = -1),
                                    rtol = FLOAT_RTOL, atol = FLOAT_ATOL )
