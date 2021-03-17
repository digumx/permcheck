"""
Defines classes representing the postcondition derived at each layer from the given precodition, and
methods for pushing the postcondition forward across a layer.
"""


import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from global_consts import SCIPY_LINPROG_METHOD


class LinearPostcond:
    """
    A `LinearPostcon`dition is a regions of the form vM + c, |v| <= b. It is thus given by three
    things, a kxn matrix `M` representing a set of k n-dimensional basis vectors, one for each row;
    a n-dimensional center point `c`, and a k-dimensional bounding vector `b`. This class simply
    acts as a struct containing all the relevant data. b should always have only positive values.

    Members

    basis   -   The M.
    center  -   The c.
    bound   -   The b.
    """
    
    def __init__(self, M : ArrayLike, c : ArrayLike, b : ArrayLike):

        # Invariant for LinearPostcond TODO remove
        assert M.ndim == 2 and c.ndim == 1 and b.ndim == 1
        assert M.shape[0] == b.shape[0] and M.shape[1] == c.shape[0]
        
        self.basis : ArrayLike = M
        self.center : ArrayLike = c
        self.bound : ArrayLike = b


def push_forward_relu_tie_class(left_cond: LinearPostcond) -> LinearPostcond:
    """
    Pushes a `LinearPostcondition` across a ReLu layer using the tie-class analysis.
    """

    ## Calculate tie classes
    
    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

    print(f"Center classes {tc_pos}, {tc_neg}")     #DEBUG

    # Bounds to use for lp
    bnds = np.array([-left_cond.bound, left_cond.bound]).transpose()
    
    # Do the tie class analysis for each group. The sign are so that the region of the lp used to
    # detect tie classes can always be written as ax <= b.
    tie_classes = []
    for tc_src, sgn in ((tc_pos, 1), (tc_neg, -1)):
        print(f"Checking group {sgn} with {tc_src} indices") #DEBUG
        
        # While there are more tie classes to be found
        while len(tc_src) > 0:
            print(f"Remaining number of neurons to classify: {len(tc_src)}")

            i = tc_src[0]
            tc_src_ = []
            tc = [i]

            # The following lp a.x <= b denotes the x for which i'th relu will have a positive or
            # negative value, depending on weather we are checking tc_pos, or tc_neg
            a = (-sgn) * left_cond.basis[np.newaxis,:,i]    # ax + b >= 0  -->  (-a)x <= b
            b = sgn * left_cond.center[np.newaxis,i]                   # ax + b <= 0  -->  ax <= -b
            #print("a, b: ", a, b) #DEBUG

            # For each other index in source of indices
            for j in tc_src[1:]:
                #print(f"Indices {i}, {j}:")  #DEBUG
                # Run solver for each pair in i,j. We find the exremal value of the j'th relu
                res = linprog(sgn * left_cond.basis[:,j], a, b, bounds = bnds,
                                    method = SCIPY_LINPROG_METHOD)
                if res.status != 0:
                    raise RuntimeError("Linear optimizer failed")
                # Analyze result.
                if res.fun >= (-sgn) * left_cond.center[j]:     
                    tc.append(j)                                # Same tie class
                    #print("Tied")   #DEBUG
                else:
                    tc_src_.append(j)                           # We will look at this again
                    #print("Untied")   #DEBUG
            
            tie_classes.append(tc)
            tc_src = tc_src_

    print(tie_classes)      # DEBUG


if __name__ == '__main__':

    #basis = np.array([  [1, 1, 0, 0],
    #                    [1, 1, 1, 0],
    #                    [1, 1, 1, 1]    ])
    #center = np.array([0, 0, 0, 0])
    #bound = np.array([1, 1, 1])
    
    #basis = np.array([  [1, 0, 0, 0],
    #                    [0, 1, 0, 0],
    #                    [0, 0, 1, 0],
    #                    [0, 0, 0, 1]    ])
    #center = np.array([2, 2, -2, -2])
    #bound = np.array([1, 1, 1, 1])
    
    #basis = np.array([  [1, 1, 0, 0],
    #                    [0, 0, 1, 0],
    #                    [0, 0, 0, 1]    ])
    #center = np.array([2, 2, -2, -2])
    #bound = np.array([1, 1, 1, 1])

    k = 500
    n = 500
    basis = np.random.rand(k, n)
    center = np.random.rand(n)
    bound = np.random.rand(k)

    
    push_forward_relu_tie_class(LinearPostcond(basis, center, bound))                
