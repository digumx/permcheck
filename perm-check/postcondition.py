"""
Defines classes representing the postcondition derived at each layer from the given precodition, and
methods for pushing the postcondition forward across a layer.
"""

from typing import List

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.linalg import svd

from global_consts import SCIPY_SVD_METHOD, FLOAT_ATOL, FLOAT_RTOL
from global_consts import REDUCE_POSTCOND_BASIS_LINEAR, REDUCE_POSTCOND_BASIS_RELU
from utils import check_parallel
from concurrency import log, init


class LinearPostcond:
    """
    A `LinearPostcon`dition is a regions of the form vM + c, |v| <= 1. It is thus given by two
    things, a kxn matrix `M` representing a set of k n-dimensional basis vectors, one for each row;
    and a n-dimensional center point `c`. This class simply acts as a struct containing all the
    relevant data.

    Members

    basis       -   The M.
    perp_basis  -   A matrix whose rows form a basis for the space perpendicular to the row rank of
                    `self.basis`, if not None
    center      -   The c.
    packed_mat  -   A matrix where the first k rows give M, and last gives c. This packed
                    representation is used to prevent copies and improve performance.
    reg_dim     -   The dimensionality of the region, that is, the dimensionality of v.
    num_neuron  -   The number of neurons this condition is over, that is, the dimensionality of c.    
    """
    
    def __init__(self, *args, perp_basis = None):
        """
        Construct a new `LinearPostcond`. If one argument is given, it is assumed to be the packed
        matrix representing the postcondition. Else, two arguments are expected, first being a matrix
        where each row is a basis, and the other being the vector c. Optionally, a third keyword
        argument `perp_basis` may be provided. If provided, it must be a matrix whose rows form the
        basis of the space perpendicular to the row span of the first matrix given.
        """
    
        if len(args) == 1:
            self.packed_mat = args[0]
            self.reg_dim : int = self.packed_mat.shape[0] - 1
            self.num_neuron : int = self.packed_mat.shape[1]
        else:
            # Invariant for LinearPostcond TODO remove
            assert args[0].ndim == 2
            assert args[1].ndim == 1
            assert args[0].shape[1] == args[1].shape[0]
            
            self.reg_dim = args[0].shape[0]
            self.num_neuron = args[0].shape[1]
            self.packed_mat : ArrayLike = np.zeros((self.reg_dim+1, self.num_neuron))
            self.packed_mat[:-1, :] = args[0]
            self.packed_mat[-1, :] = args[1]
            
        self.basis : ArrayLike = self.packed_mat[:-1, :]
        self.center : ArrayLike = self.packed_mat[-1, :]
        
        # Fill in the perpendicular space
        self.perp_basis = perp_basis
        
    def __repr__(self):
        return "LinearPostcond( packed_mat = {0}, perp_basis = {1}, num_neuron = {2} )".format(
                repr(self.packed_mat), repr(self.perp_basis), repr(self.num_neuron))


def optimize_postcond_basis(bss: ArrayLike, rnk = None) -> ArrayLike:
    """
    Given `bss` as the generating set of vectors for a postcondition, returns an optimized minimal
    orthogonal basis that captures all the behavior of the given postcondition. Thus, replacing
    the "basis" of a postcondition to the output of this function on the "basis" captures all
    behavior and reduces the dimension of the postcondition. If `rnk` is given, it is assumed to be
    the rank of `bss`.
    
    NOTE: This destroys the input bss.
    """

    log("Optimizing postcond basis via SVD")
    
    #u, s, v = svd(bss, overwrite_a=True, check_finite=False, lapack_driver=SCIPY_SVD_METHOD)
    u, s, v = svd(bss, overwrite_a=False, check_finite=True, lapack_driver=SCIPY_SVD_METHOD) #DEBUG
    
    if np.all( s < FLOAT_ATOL ):    # TODO This can happen if image is a single point, allow this
        log("All bases are close to 0")
        raise RuntimeError("Attempted to optimize Zero Basis")
    
    # Find the rank of bss
    cnum = FLOAT_ATOL * np.max(s)
    rnk = rnk if rnk is not None else np.amax(np.where(s >= cnum))+1
    u = u[:, :rnk]                          # Trim u, s, v.
    s = s[:rnk]
    p = v[rnk:, :]
    v = v[:rnk, :]
    b = np.sum(np.absolute(u*s), axis=0)    # New bounds
    
    if np.any( b < FLOAT_ATOL ): #DEBUG
        log("Bounds are close to 0")
        raise RuntimeError()
    
    return v * b[:, np.newaxis], p

    

def push_forward_postcond_relu(left_cond: LinearPostcond,
                                optimize_basis : bool = REDUCE_POSTCOND_BASIS_RELU) -> LinearPostcond:
    """
    Pushes forward a `LinearPostcond` across a relu via tie class analysis. Returns a LinearPostcond
    for the right side of the ReLU. If `optimize_basis is true
    """

    # First, we use tie class analysis to build a list of basis vectors for each class.

    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

    out_list = []       # A list containing (col_indices, basis, row_start, row_end) per tie class
    n_basis = 0         # Number of output basis vectors
    
    log("Performing Tie Class analysis")
    
    # Do the tie class analysis for each group. The sign denotes which region we are operating on.
    for tc_src, sgn in ((tc_pos, 1), (tc_neg, -1)):
        
        log("Classifying columns with center sign {0}".format(sgn))
        
        # While there are more tie classes to be found
        while len(tc_src) > 0:

            log("{0} columns remaining".format(len(tc_src)))

            i = tc_src[0]
            tc_src_ = []
            tc = [i]
            rnk_1 = True                    # Is the tie class of i guaranteed to have 1 basis

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
                    rnk_1 = False           # Then we cannot argue that class of i must have 1 basis
                    tc.append(j)
                # Else, check if joint system is one dimensional
                elif np.all( check_parallel(i_vec, j_vec, anti_are_parallel = False) ):
                    tc.append(j)
                # Else, cannot be in same tie classes, look at it again in the future
                else:
                    tc_src_.append(j)
            
            
            # Once we have a tie class, calculate a minimal basis representing it, and add relevant
            # info to out_basis
            
            # Early shortcut if tie class has only one element
            if len(tc) == 1:
                v = np.array([[ np.sum(np.absolute(left_cond.basis[:,tc[0]])) ]])
                out_list.append((tc, v, n_basis, n_basis+1))
                n_basis += 1
            
            # Else just add the generating vectors in
            else:
                tcg = left_cond.basis[:,tc]
                rnk = tcg.shape[0]
                out_list.append((tc, tcg, n_basis, n_basis+rnk))
                n_basis += rnk
               
            tc_src = tc_src_                            # Update list of unclassified indices
        
        
    # Copy data from out_list into correct indices to create basis matrix
    basis = np.zeros((n_basis, left_cond.num_neuron))
    for tc, bs, rb, eb in out_list:
        basis[rb:eb, tc] = bs
        
    # New center is relu of old center
    center = np.copy(left_cond.center)
    center[np.where(left_cond.center < 0)] = 0
 
    # Reduce basis if flag set
    if optimize_postcond_basis:
        b, p = optimize_postcond_basis(basis)
        return LinearPostcond(b, center, perp_basis=p)
    
    return LinearPostcond(basis, center)


def push_forward_postcond_linear(postc: LinearPostcond, weights: ArrayLike, bias: ArrayLike,
                                optimize_basis = REDUCE_POSTCOND_BASIS_LINEAR) -> LinearPostcond:
    """
    Push forward a linear postcondition across a linear layer layer of a Neural Network. The
    arguments are:
    
    0.  postc                   -   The postcondition before the relu of the of the i-th layer.
    1.  weights                 -   The weights of the (i+1)-th layer's linear transform
    2.  biases                  -   The biases of the (i+1)-th layer's linear transform
    3.  optimize_postcond_basis -   If true, optimizes the basis of the returned postcondition
    """
    # Push forward across linear layer.
    post_spn = postc.basis @ weights
    post_center = postc.center @ weights + bias
    
    # Optimise span to basis
    if optimize_basis:
        post_basis, perp_basis = optimize_postcond_basis(post_spn)
        return LinearPostcond(post_basis, post_center, perp_basis = perp_basis)
    
    # Or just return as is
    return LinearPostcond(post_spn, post_center)    
    
    




if __name__ == '__main__':


    from timeit import timeit
    import random
    import sys
    
    from debug import rand_sparce_matrix
   
    init(None)
   
    if sys.argv[1] == "unit":
    
        if sys.argv[2] == "1":
            
            postc = LinearPostcond(np.array([[1000, -1000, 1000, -1000, 1000, -1000, 1000, -1000],
                                            [-1000, 1000, -1000, 1000, -1000, 1000, -1000, 1000]]),
                                   np.array([0, 0, -1, -1, 0, 0, -1, -1]))
            w = np.array([  [ 1, 0, 0, 0],
                            [ 0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1],
                            [ 0, 0,-1, 0],
                            [ 0, 0, 0,-1] ])
            b = np.array([ 0, 0, 0, 0 ])
            pf = push_forward_postcond(postc, w, b)
            print("Basis: {0}, Center: {1}".format(pf.basis, pf.center))
        
        if sys.argv[2] == "2":
            
            postc = LinearPostcond(np.array([[1, -1, 1, -1],
                                            [-1, 1, -1, 1]]),
                                   np.array([0, 0, 0, 0]))
            w = np.array([  [ 1, 0, 0, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1] ])
            b = np.array([ 0, 0, 0, 0 ])
            pf = push_forward_postcond(postc, w, b)
            print("Basis: {0}, Center: {1}".format(pf.basis, pf.center))
    
    elif sys.argv[1] == "fuzz":

        k = 200
        n = 200
        n_ = 120
        cenvar = 10
        biavar = 10
        p0 = random.random() * 0.1
        p1 = random.random() * 0.15
        n_run = 1000
        
        t = 0

        basis, center, weights, bias = None, None, None, None
        
        def fail_dump():
            global n, k, basis, center, tc1, tc2
            # Dump basis, center and tie classes found to log if methods do not match.
            data = {}
            data['basis'] = basis.tolist()
            data['center'] = center.tolist()
            data['weights'] = weights.tolist()
            data['bias'] = bias.tolist()
            with open(sys.argv[1], 'w') as logf:
                logf.write(str(data))
            
        def run_pf():
            global tc1, basis, center
            log("Running bound based push forward")
            push_forward_postcond(LinearPostcond(basis, center), weights, bias)
        
        if len(sys.argv) >= 3 and sys.argv[2] == "checklog":
            with open(sys.argv[1]) as log:
                data = eval(log.read())
                basis = np.array(data['basis'])
                center = np.array(data['center'])
                weights = np.array(data['weights'])
                bias = np.array(data['bias'])
                
                log("Running pushforward")
                try:
                    t += timeit(run_pf, number=1)
                except Exception as e:
                    fail_dump()
                    raise e
                
                exit()
                
        
        for i in range(n_run):
            log(f"Run {i} of {n_run}")

            log("Generating data")
            basis = rand_sparce_matrix(k,n,p0)
            center = (np.random.rand(n) - 0.5) * cenvar
            
            weights = rand_sparce_matrix(n,n_,p0)
            bias = (np.random.rand(n_) - 0.5) * biavar
            
            log("Running pushforward")
            try:
                t += timeit(run_pf, number=1)
            except Exception as e:
                fail_dump()
                raise e

        t /= n_run
        log(f"The average time for pushforward is {t}")
        log(f"There were {n} relu neurons and the left space was {k} dimensional")
