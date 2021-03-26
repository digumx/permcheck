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
        matrix representing the postcondition. Else, two arguments are expected, first being a matrix
        where each vector is a basis, and the other being the vector c.
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



def push_forward_postcond_relu(left_cond: LinearPostcond) -> List[ArrayLike]:
    """
    Pushes forward a `LinearPostcond` across a relu via tie class analysis. Returns the
    `LinearPostcond` to the right side of the relu.
    """

    # First, we use tie class analysis to build a list of basis vectors for each class.

    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

    out_list = []       # A list containing (col_indices, basis, row_start, row_end) per tie class
    n_basis = 0         # Number of output basis vectors
    
    # Do the tie class analysis for each group. The sign denotes which region we are operating on.
    for tc_src, sgn in ((tc_pos, 1), (tc_neg, -1)):
        #print(f"Checking group {sgn}") #DEBUG
        
        # While there are more tie classes to be found
        while len(tc_src) > 0:
            #print(f"Remaining number of neurons to classify: {len(tc_src)}      ", end='\r')#DEBUG

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
                elif np.allclose( np.inner(i_vec, j_vec), np.linalg.norm(i_vec) * np.linalg.norm(j_vec),
                                    rtol = FLOAT_RTOL, atol = FLOAT_ATOL ):
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
            
            else:
                tcg = left_cond.basis[:,tc]             # The generating vectors for the tie class
                u, s, v = svd(tcg, overwrite_a=True, check_finite=False,
                                                lapack_driver=SCIPY_SVD_METHOD)
                
                # Find the rank of tcg
                rnk = 1 if rnk_1 else np.amax(np.where(np.absolute(s) >= FLOAT_ATOL))+1
                u = u[:, :rnk]                          # Trim u, s, v.
                s = s[:rnk]
                v = v[:rnk, :]
                b = np.sum(np.absolute(u*s), axis=0)    # New bounds
                
                # Add in required info to list of stuff
                out_list.append((tc, v / b[:,np.newaxis], n_basis, n_basis+rnk))
                n_basis += rnk
               
            tc_src = tc_src_                            # Update list of unclassified indices
        
        
    # Copy data from out_list into correct indices to create basis matrix
    basis = np.zeros((n_basis, left_cond.num_neuron))
    for tc, bs, rb, eb in out_list:
        basis[rb:eb, tc] = bs
        
    # New center is relu of old center
    center = np.copy(left_cond.center)
    center[np.where(left_cond.center < 0)] = 0
    
    return LinearPostcond(basis, center)


def push_forward_postcond(postc: LinearPostcond, weights: ArrayLike, bias: ArrayLike) \
                                                                            -> LinearPostcond:
    """
    Push forward a linear postcondition across a layer of a Neural Network. Assuming the given
    postcondition to be on the values entering the relu of the i-th layer, returns a postcondition
    on the values entering the relu of the (i+1)-th layer. The arguments are:
    
    0.  postc   -   The postcondition before the relu of the of the i-th layer.
    1.  weights -   The weights of the (i+1)-th layer's linear transform
    2.  biases  -   The biases of the (i+1)-th layer's linear transform
    """
    
    # Push forward across relu
    relupc = push_forward_postcond_relu(postc)
    
    # Push forward across linear layer.
    post_spn = relupc.basis @ weights
    post_center = relupc.center @ weights + bias
    
    # Optimise span to basisl
    _, s, v = svd(post_spn, overwrite_a=True, check_finite=False, lapack_driver=SCIPY_SVD_METHOD)
    rnk = np.amax(np.where(np.absolute(s) >= FLOAT_ATOL)) + 1
    post_basis = v[:rnk, :]                             # Trim u to get basis
    
    return LinearPostcond(post_basis, post_center)
    




if __name__ == '__main__':


    from timeit import timeit
    import random
    import sys

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
        with open(sys.argv[1], 'w') as log:
            log.write(str(data))
        
    def run_pf():
        global tc1, basis, center
        print("Running bound based push forward")
        push_forward_postcond(LinearPostcond(basis, center), weights, bias)
    
    if len(sys.argv) >= 3 and sys.argv[2] == "checklog":
        with open(sys.argv[1]) as log:
            data = eval(log.read())
            basis = np.array(data['basis'])
            center = np.array(data['center'])
            weights = np.array(data['weights'])
            bias = np.array(data['bias'])
            
            print("Running pushforward")
            try:
                t += timeit(run_pf, number=1)
            except Exception as e:
                fail_dump()
                raise e
            
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
        
        weights = np.random.rand(n,n_)
        neg_idx = np.where(weights < p0)
        zer_idx = np.where(np.logical_and(weights >= p0, weights <= (1-p0)))
        pos_idx = np.where(weights > 1-p0)
        weights[neg_idx] -= p0
        weights[neg_idx] /= p0
        weights[zer_idx] *= 0
        weights[pos_idx] -= 1-p0
        weights[pos_idx] /= 1-p0
        bias = (np.random.rand(n_) - 0.5) * biavar
        
        print("Running pushforward")
        try:
            t += timeit(run_pf, number=1)
        except Exception as e:
            fail_dump()
            raise e

    t /= n_run
    print(f"The average time for pushforward is {t}")
    print(f"There were {n} relu neurons and the left space was {k} dimensional")
