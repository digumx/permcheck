"""
Defines classes representing the postcondition derived at each layer from the given precodition, and
methods for pushing the postcondition forward across a layer.
"""

from typing import List

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from global_consts import SCIPY_LINPROG_METHOD, FLOAT_ATOL, FLOAT_RTOL


class LinearPostcond:
    """
    A `LinearPostcon`dition is a regions of the form vM + c, |v| <= 1. It is thus given by two
    things, a kxn matrix `M` representing a set of k n-dimensional basis vectors, one for each row;
    and a n-dimensional center point `c`. This class simply acts as a struct containing all the
    relevant data.

    Members

    basis       -   The M.
    center      -   The c.
    packed_mat  -   A matrix where the first k rows give M, and last gives c. This packed
                    representation is used to prevent copies and improve performance.
    reg_dim     -   The dimensionality of the region, that is, the dimensionality of v.
    num_neuron  -   The number of neurons this condition is over, that is, the dimensionality of c.    
    """
    
    def __init__(self, *args):
        """
        Construct a new `LinearPostcond`. If one argument is given, it is assumed to be the packed
        matrix representing the postcondition. Else, two arguments are expected, one being a matrix
        where each vector is a basis, and the other being the vector c.
        """
    
        if len(args) == 1:
            self.packed_mat = args[0]
        else:
            # Invariant for LinearPostcond TODO remove
            assert args[0].ndim == 2
            assert args[1].ndim == 1
            assert args[0].shape[1] == args[1].shape[0]
            self.packed_mat : ArrayLike = np.zeros((k+1, n))
            self.packed_mat[:-1, :] = args[0]
            self.packed_mat[-1, :] = args[1]
            
        self.basis : ArrayLike = self.packed_mat[:-1, :]
        self.center : ArrayLike = self.packed_mat[-1, :]
        self.reg_dim : int = self.packed_mat.shape[0] - 1
        self.num_neuron : int = self.packed_mat.shape[1]



def tie_classify_lp(left_cond: LinearPostcond) -> List[ArrayLike]:
    """
    Given a linear postcondition, uses a linear program solver to generate a list of tie classes.
    The tie classes are returned as a list of numpy index arrays, each element in the list refering
    to a tie class.
    """

    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

    # Bounds to use for lp
    bnds = np.array([-np.ones(left_cond.reg_dim), np.ones(left_cond.reg_dim)]).transpose()
    
    # Do the tie class analysis for each group. The sign are so that the region of the lp used to
    # detect tie classes can always be written as ax <= b.
    tie_classes = []
    for tc_src, sgn in ((tc_pos, 1), (tc_neg, -1)):
        print(f"Checking group {sgn}") #DEBUG
        
        # While there are more tie classes to be found
        while len(tc_src) > 0:
            print(f"Remaining number of neurons to classify: {len(tc_src)}      ", end='\r')

            i = tc_src[0]
            tc_src_ = []
            tc = [i]

            # The following lp a.x <= b denotes the x for which i'th relu will have a positive or
            # negative value, depending on weather we are checking tc_pos, or tc_neg
            a = (-sgn) * left_cond.basis[np.newaxis,:,i]            # ax + b >= 0  -->  (-a)x <= b
            b = sgn * left_cond.center[np.newaxis,i]                # ax + b <= 0  -->  ax <= -b
            #print("a, b: ", a, b) #DEBUG

            # For each other index in source of indices
            for j in tc_src[1:]:
                
                # Run solver for each pair in i,j. We find the exremal value of the j'th relu
                res1 = linprog(sgn * left_cond.basis[:,j], a, b, bounds = bnds,
                                    method = SCIPY_LINPROG_METHOD)
                
                # Now reverse and set j to a quadrant and check if i is out of that qudrant
                a_ = (-sgn) * left_cond.basis[np.newaxis,:,j]           # ax + b >= 0  -->  (-a)x <= b
                b_ = sgn * left_cond.center[np.newaxis,j]               # ax + b <= 0  -->  ax <= -b
                
                # And, We find the exremal value of the i'th relu
                res2 = linprog(sgn * left_cond.basis[:,i], a_, b_, bounds = bnds,
                                    method = SCIPY_LINPROG_METHOD)
                
                if res1.status != 0 or res2.status != 0:
                    raise RuntimeError(f"Linear optimizer failed with {res1.staus}, {res2.status}")
                # Analyze result.
                if res1.fun >= (-sgn) * left_cond.center[j] \
                    and res2.fun >= (-sgn) * left_cond.center[i]:     
                    tc.append(j)                                # Same tie class
                    #print("Tied")   #DEBUG
                else:
                    tc_src_.append(j)                           # We will look at this again
                    #print("Untied")   #DEBUG
            
            tie_classes.append(np.array(tc))
            tc_src = tc_src_
        print()

    return tie_classes


def tie_classify_bound(left_cond: LinearPostcond) -> List[ArrayLike]:
    """
    Given a linear postcondition, uses bound analysis to generate a list of tie classes.  The tie
    classes are returned as a list of numpy index arrays, each element in the list refering to a tie
    class.
    """

    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

    # Do the tie class analysis for each group. The sign denotes which region we are operating on.
    tie_classes = []
    for tc_src, sgn in ((tc_pos, 1), (tc_neg, -1)):
        print(f"Checking group {sgn}") #DEBUG
        
        # While there are more tie classes to be found
        while len(tc_src) > 0:
            print(f"Remaining number of neurons to classify: {len(tc_src)}      ", end='\r')

            i = tc_src[0]
            tc_src_ = []
            tc = [i]

            i_cent = left_cond.center[i]
            i_col = left_cond.basis[:,i]
            i_vec = left_cond.packed_mat[:,i]

            # Check if i neuron's value is bounded
            i_bdd = sgn * i_cent - np.sum(np.absolute(i_col)) >= 0

            # For each other index in source of indices
            for j in tc_src[1:]:
                j_cent = left_cond.center[j]
                j_col = left_cond.basis[:,j]
                j_vec = left_cond.packed_mat[:,j]
                
                # If both bounded, both are in same tie class
                if i_bdd and sgn * j_cent - np.sum(np.absolute(j_col)) >= 0:
                    tc.append(j)
                # Else, check if joint system is one dimensional
                elif np.allclose( np.inner(i_vec, j_vec), np.linalg.norm(i_vec) * np.linalg.norm(j_vec),
                                    rtol = FLOAT_RTOL, atol = FLOAT_ATOL ):
                    tc.append(j)
                # Else, cannot be in same tie classes
                else:
                    tc_src_.append(j)
            
            tie_classes.append(np.array(tc))
            tc_src = tc_src_
        print()

    return tie_classes
                 

    


# TODO Complete
def push_forward_relu_tie_class(left_cond: LinearPostcond) -> LinearPostcond:
    """
    Pushes a `LinearPostcondition` across a ReLu layer using the tie-class analysis.
    """

    ## Calculate tie classes
    
    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

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
        print()

    print(tie_classes)      # DEBUG


if __name__ == '__main__':


    from timeit import timeit

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

    import random
    import sys

    k = 200
    n = 200
    invar = 10
    cenvar = 10
    p0 = random.random() * 0.150
    n_run = 1000
    
    t_1, t_2 = 0,0

    basis, center, tc1, tc2 = None, None, None, None
    
    def fail_dump():
        global n, k, basis, center, tc1, tc2
        # Dump basis, center and tie classes found to log if methods do not match.
        data = {}
        data['basis'] = [ [ c for c in b ] for b in basis ]
        data['center'] = [ c for c in center ]
        data['tc1'] = [ [ tm for tm in tc ] for tc in tc1 ]
        data['tc2'] = [ [ tm for tm in tc ] for tc in tc2 ]
        with open(sys.argv[1], 'w') as log:
            log.write(str(data))
        
    def run_tc1():
        global tc1, basis, center
        print("Running lp based tie classifier")
        tc1 = tie_classify_lp(LinearPostcond(basis, center))
    
    def run_tc2():
        global tc2, basis, center   
        print("Running bounds based tie classifier")
        tc2 = tie_classify_bound(LinearPostcond(basis, center))
   
    def check_tc_same(tc1, tc2):
        tc2_ = tc2[:]
        if len(tc1) != len(tc2):
            print("FAILED")
            fail_dump()
            exit()
        for c1 in tc1:
            i0 = -1
            for i, c2 in enumerate(tc2_):
                if set(c1.tolist()) == set(c2.tolist()):
                    i0 = i
                    break
            if i0 == -1:
                print("FAILED")
                fail_dump()
                exit()
            tc2_ = (tc2_[:i0] + tc2_[i0+1:]) if i0 < len(tc2_) - 1 else tc2_[:-1]
        
        if len(tc2_) > 0:
            print("FAILED")
            fail_dump()
            exit()
    
    if len(sys.argv) >= 3 and sys.argv[2] == 'checklog':
        with open(sys.argv[1], 'r') as log:
            data = eval(log.read())
            basis = np.array(data['basis'])
            center = np.array(data['center'])
            
            t_1 += timeit(run_tc1, number=1)
            t_2 += timeit(run_tc2, number=1)
        
            print("Checking equality")
            check_tc_same(tc1, tc2)
            print("SUCCESS")
            
            exit()
    
    for i in range(n_run):
        print(f"Run {i} of {n_run}")

        print("Generating data")
        basis = np.random.rand(k,n)
        neg_idx = np.where(basis < p0)
        zer_idx = np.where(np.logical_and(basis >= p0, basis <= (1-p0)))
        pos_idx = np.where(basis > 1-p0)
        basis[neg_idx] -= p0
        basis[neg_idx] /= p0
        basis[zer_idx] *= 0
        basis[pos_idx] -= 1-p0
        basis[pos_idx] /= 1-p0
        center = (np.random.rand(n) - 0.5) * cenvar
        

        t_1 += timeit(run_tc1, number=1)
        t_2 += timeit(run_tc2, number=1)

        print("Checking equality")
        check_tc_same(tc1, tc2)
        print("SUCCESS")

    t_1 /= n_run
    t_2 /= n_run
    print(f"The time for LP using {SCIPY_LINPROG_METHOD} is {t_1}, and for the other method is {t_2}")
    print(f"There were {n} relu neurons and the left space was {k} dimensional")
