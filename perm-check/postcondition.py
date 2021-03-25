"""
Defines classes representing the postcondition derived at each layer from the given precodition, and
methods for pushing the postcondition forward across a layer.
"""

from typing import List

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from numpy.linalg import svd

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
        matrix representing the postcondition. Else, two arguments are expected, first being a matrix
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


# TODO Refactor this function out
def tie_classify_bound(left_cond: LinearPostcond) -> List[ArrayLike]:
    """
    Given a linear postcondition, uses bound analysis to generate a list of tie classes. For each
    tie class, it also checks if the tie class contains components which take both positive and
    negative values. Returns two lists. The first one has a numpy array of indices in each tie
    class, and the second has boolean values for the respective tie classes denoting weather the
    components in the class can have both signs in the left space.
    """

    # The center point defines some of the cuts for the tie class - things in the same tie class
    # should have same sign in center.
    tc_pos = np.where(left_cond.center >= 0)[0]
    tc_neg = np.where(left_cond.center < 0)[0]

    # Do the tie class analysis for each group. The sign denotes which region we are operating on.
    tie_classes = []
    pos_and_neg = []
    for tc_src, sgn in ((tc_pos, 1), (tc_neg, -1)):
        print(f"Checking group {sgn}") #DEBUG
        
        # While there are more tie classes to be found
        while len(tc_src) > 0:
            print(f"Remaining number of neurons to classify: {len(tc_src)}      ", end='\r')

            i = tc_src[0]
            tc_src_ = []
            tc = [i]
            pan = True

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
                    pan = False
                    tc.append(j)
                # Else, check if joint system is one dimensional
                elif np.allclose( np.inner(i_vec, j_vec), np.linalg.norm(i_vec) * np.linalg.norm(j_vec),
                                    rtol = FLOAT_RTOL, atol = FLOAT_ATOL ):
                    tc.append(j)
                # Else, cannot be in same tie classes
                else:
                    tc_src_.append(j)
            
            tie_classes.append(np.array(tc))
            pos_and_neg.append(pan)
            tc_src = tc_src_
        print()

    return tie_classes, pos_and_neg
                 

def push_forward_postcond_relu(left_cond: LinearPostcond) -> List[ArrayLike]:
    """
    Pushes forward a `LinearPostcond` across a relu via tie class analysis. Returns the
    `LinearPostcond` to the right side of the relu.
    """
    
    # Use one of the above methods to calculate the tie classes. Bounds is significantly faster.
    tie_classes, unit_rank = tie_classify_bound(left_cond)
    
    out_list = []
    n_basis = 0
    
    # For each tie class, chose a minimal orthogonal basis representing the right space of the tie
    # class.
    for tc, urg in zip(tie_classes, unit_rank):
        
        # Early shortcut if tie class has only one element
        if len(tc) == 1:
            v = np.array([[ np.sum(np.absolute(left_cond.basis[:,tc[0]])) ]])
            out_list.append((tc, v, n_basis, n_basis+1))
            n_basis += 1
            continue
        
        tcg = left_cond.basis[:,tc]             # The generating vectors for the tie class
        #print(f"tcg {tcg.shape}") #DEBUG
        u, s, v = svd(tcg)                      # Perform svd
        #print(f"u {u.shape} s {s.shape} v {v.shape}") #DEBUG
        
        # Find the rank of tcg
        rnk = 1 if urg else np.amax(np.where(np.absolute(s) >= FLOAT_ATOL))+1
        u = u[:, :rnk]                          # Trim u, s, v.
        s = s[:rnk]
        v = v[:rnk, :]
        b = np.sum(np.absolute(u*s), axis=0)    # New bounds
        
        #print(f"u {u.shape} s {s.shape} v {v.shape}") #DEBUG
        
        # Add in required info to list of stuff
        out_list.append((tc, v / b[:,np.newaxis], n_basis, n_basis+rnk))
        n_basis += rnk
        
    # Create basis matrix
    basis = np.zeros((n_basis, left_cond.num_neuron))
    for tc, bs, rb, eb in out_list:
        basis[rb:eb, tc] = bs
        
    # New center is relu of old center
    center = np.copy(left_cond.center)
    center[np.where(left_cond.center < 0)] = 0
    
    return LinearPostcond(basis, center)


if __name__ == '__main__':


    from timeit import timeit
    import random
    import sys

    k = 200
    n = 200
    invar = 10
    cenvar = 10
    p0 = random.random() * 0.1
    n_run = 1000
    
    t = 0

    basis, center = None, None
    
    def fail_dump():
        global n, k, basis, center, tc1, tc2
        # Dump basis, center and tie classes found to log if methods do not match.
        data = {}
        data['basis'] = basis.tolist()
        data['center'] = center.tolist()
        with open(sys.argv[1], 'w') as log:
            log.write(str(data))
        
    def run_pf():
        global tc1, basis, center
        print("Running bound based push forward")
        push_forward_postcond_relu(LinearPostcond(basis, center))
    
    if len(sys.argv) >= 3 and sys.argv[2] == "checklog":
        with open(sys.argv[1]) as log:
            data = eval(log.read())
            basis = np.array(data['basis'])
            center = np.array(data['center'])
            
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
        
        print("Running pushforward")
        try:
            t += timeit(run_pf, number=1)
        except Exception as e:
            fail_dump()
            raise e

    t /= n_run
    print(f"The average time for pushforward is {t}")
    print(f"There were {n} relu neurons and the left space was {k} dimensional")
