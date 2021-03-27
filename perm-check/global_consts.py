"""
Defines global constant values used by all other files. These include various configuration settings,
constants, etc. 
"""

from enum import Enum





"""
Configuration for various scipy linear arlgebra algos.
"""


class ScipyLinprogMethod(Enum):
    """
    This enum lists the possible values the `scipy.optimize.linprog`'s `method` parameter can take,
    and refer to the various linear program solvers implemented in scipy. Refer to 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    NOTE:   The HIGHS_AUTO chooses between the dual revised simplex and interior point method
            solvers implemented in HIGHS automatically, based on some heuristics. Choosing one of
            them manually based on benchmarks may be better.
    """
    HIGHS_DS    = 'highs-ds'        # High perf dual revised simplex
    HIGHS_IMP   = 'highs-ipm'       # High perf interior point method
    HIGHS_AUTO  = 'highs'           # Auto choose between two highs (don't use)
    DEFAULT     = 'interior-point'  # Default, robust interior point method
    ACC_SIMPLEX = 'revised simplex' # A more accurate legacy simplex
    LEG_SIMPLEX = 'simplex'         # The legacy simplex implementation

SCIPY_LINPROG_METHOD = ScipyLinprogMethod.HIGHS_DS.value    # Must be one of the above


class ScipySVDMethod(Enum):
    """
    What method to use for the SVD decomposition for a matrix. Documentation and SO suggests that
    GESDD should be fastest.
    """
    GESDD   = 'gesdd'       # New Divide and conquer method
    GESVD   = 'gesvd'       # Old rectangular method, used by matlab

SCIPY_SVD_METHOD = ScipySVDMethod.GESDD.value





"""
Numerical constants, for checking closeness of floating point numbers. Two floats a and b are
considered 'close' if |b - a| < abs(b) * FLOAT_RTOL + FLOAT_ATOL. The values used in numpy are given
at: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html, these may be good values
""" # TODO scrutinize use
FLOAT_RTOL = 1e-5
FLOAT_ATOL = 1e-8




"""
Configuration relating to the push forward of postconditions.
"""

"""
If set tu true, when postconditions are pushed forward, redundant generating vectors are discarded
and only a minimal basis is maintained. This reduces the number of vectors to keep track of and
improves performance of future push forwards and inclusion checks, but introduces unreachable
behavior and weakens the postcondition.
"""
REDUCE_POSTCOND_BASIS = True