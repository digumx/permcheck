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
    HIGHS_IPM   = 'highs-ipm'       # High perf interior point method
    HIGHS_AUTO  = 'highs'           # Auto choose between two highs (don't use)
    DEFAULT     = 'interior-point'  # Default, robust interior point method
    ACC_SIMPLEX = 'revised simplex' # A more accurate legacy simplex
    LEG_SIMPLEX = 'simplex'         # The legacy simplex implementation

SCIPY_LINPROG_METHOD = ScipyLinprogMethod.HIGHS_IPM.value    # Must be one of the above


class ScipySVDMethod(Enum):
    """
    What method to use for the SVD decomposition for a matrix. Documentation and SO suggests that
    GESDD should be fastest.
    """
    GESDD   = 'gesdd'       # New Divide and conquer method
    GESVD   = 'gesvd'       # Old rectangular method, used by matlab

SCIPY_SVD_METHOD = ScipySVDMethod.GESDD.value

class ScipyLstsqMethod(Enum):
    """
    The Lapack driver Scipy should use in least-squares
    """
    GELSD   = 'gelsd'       # A good choice
    GELSY   = 'gelsy'       # May be faster for some problems
    GELSS   = 'gelss'       # Legacy
    
SCIPY_LSTSQ_METHOD = ScipyLstsqMethod.GELSD.value

class NumpySortMethod(Enum):
    """
    What method is used to sort arrays
    """
    QUICK   = 'quicksort'   # Fastest in terms of implementation, not complexity
    MERGE   = 'mergesort'   # Better complexity, but slower, also legacy
    HEAP    = 'heapsort'    # Same complexity as merge, but slower, according to docs
    
NUMPY_SORT_METHOD = NumpySortMethod.HEAP.value






"""
Numerical constants, for checking closeness of floating point numbers. Two floats a and b are
considered 'close' if |b - a| < abs(b) * FLOAT_RTOL + FLOAT_ATOL. The values used in numpy are given
at: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html, these may be good values
""" # TODO scrutinize use
FLOAT_RTOL = 1e-5
FLOAT_ATOL = 1e-8




""" Configuration relating to the push forward of postconditions.  """

"""
If set tu true, when postconditions are pushed forward, redundant generating vectors are discarded
and only a minimal basis is maintained. This reduces the number of vectors to keep track of and
improves performance of future push forwards and inclusion checks, but introduces unreachable
behavior and weakens the postcondition. We have two seperate flags for the pushforward through the
linear and relu layers respectively.
"""
REDUCE_POSTCOND_BASIS_LINEAR = True
REDUCE_POSTCOND_BASIS_RELU = False




""" Configuration for concurrency """

"""
Should multiprocess based parallelism be used?
"""
USE_MP = True

"""
The method used to spawn the worker processes, if None, the default method or the one set globally
by the parent process is used. Default is "spawn" for best portabitily and flexibility.
"""
MP_START_METHOD = "spawn"

"""
Number of workers to use. The actual number of processes will be more than this, but all the extra
processes will be doing IO and organization, so should not hold up the CPU.
"""
MP_NUM_WORKER = 10




""" Configuration for the pullback of potential counterexamples """

"""
A multiplier for the number of random samples to pick from the space of alpha values.
"""
CEX_PULLBACK_SAMPLES_SCALE = 100

"""
The default number of candidate pullbacks to return per layer. TODO Basically not used 
"""
CEX_PULLBACK_NUM_PULLBACKS = 5

"""
The default multiplier for the number of cexes candidates to check when inclusion fails. The actual
number is calculated as this times the number of neurons in the DNN.
"""
CEX_PULLBACK_NUM_TOTAL_CANDIDATES_MULT = 20
