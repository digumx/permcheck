"""
Defines functions for checking weather a computed postcondition is included within a computed
precondition.
"""


from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.linalg import null_space

from postcondition import LinearPostcond
from precondition import DisLinearPrecond, NegSideType
from global_consts import REDUCE_POSTCOND_BASIS, FLOAT_RTOL, FLOAT_ATOL, SCIPY_LINPROG_METHOD
from utils import check_parallel

MbFloat = Union[ None, float ]

def check_lp_inclusion(inner_ub_a: ArrayLike, inner_ub_b: ArrayLike, 
                        inner_eq_a: ArrayLike, inner_eq_b: ArrayLike, 
                        outer_ub_a: ArrayLike, outer_ub_b: ArrayLike, 
                        inner_bounds: Union[ list[ tuple[ MbFloat, MbFloat ] ], tuple[ MbFloat, MbFloat ]], 
                        n_cex: int, l_cex: list[ArrayLike]) -> None:
    """
    Checks weather the region { x : inner_ub_a @ x <= inner_ub_b, inner_eq_a @ x == inner_eq_b ,
    bounds[0] <= x <= bounds[1]} is contained within the region { x : outer_ub_a @ x <= outer_ub_b }
    via a number of linear program calls. Adds counterexamples found to `l_cex`, until the size of
    `l_cex` reaches `n_cex`, at which point it returns.
    """
    
    # Loop over all outer constraints, optimize for max value.
    i = 0 #DEBUG
    for c, b in zip(outer_ub_a, outer_ub_b):
        print(f"Checked constraint {i} of {len(outer_ub_a)}, found {len(l_cex)} cexes")
        i += 1
        
        res = linprog(-c, inner_ub_a, inner_ub_b, inner_eq_a, inner_eq_b, bounds = inner_bounds,
                method = SCIPY_LINPROG_METHOD)      # Call optimizer
        
        if res.status == 2:    # Inner is infeasable
            print("Infeasable inner") #DEBUG
            return
        
        elif res.status == 0:
            if -res.fun > b:    # Out of bounds, found cex
                l_cex.append(res.x)
                if len(l_cex) >= n_cex:
                    break
        
        else:
            raise RuntimeError("Optimizer returned bad status: {0}".format(res.status))
    
    print(f"Lp check leaves with {len(l_cex)} cexes")


def check_inclusion( postc: LinearPostcond, prec: DisLinearPrecond, n_cex : int = 1) \
                    -> list[ArrayLike]:
    """
    Check if given postcondition `postc` is included within the precondition `prec`. Returns a list
    of upto `n_cex` counterxamples with no repetitions, or an empty list if inclusion is satisified.
    Note returned list may contain duplicate elements.
    """
    assert postc.num_neuron == prec.num_neuron
   
    lcex = []
   
    # Get the LP for the postcondition. This works only if the postcondition has orthogonal bases
    assert REDUCE_POSTCOND_BASIS    #TODO implement other case
    
    # Calculate the bounds.
    ub_a = np.concatenate( (postc.basis, -postc.basis), axis=0 )
    t1 = postc.basis @ postc.center
    t2 = np.einsum("...i,...i->...", postc.basis, postc.basis)
    ub_b = np.concatenate( (t2 + t1, t2 - t1), axis=0 )
    
    
    # Equality should say that all components perpendicular to basis should have same value as
    # center
    if postc.perp_basis is not None:
        eq_a = postc.perp_basis
    else:
        eq_a = np.transpose( null_space(postc.basis, rcond=FLOAT_RTOL) )
    eq_b = eq_a @ postc.center
        
    # Check if postcondition is entirely within the positive region, or entirely negative.
    axvar = np.sum( np.absolute( postc.basis ), axis = 0 )  # Ub - Lb per axis
    axub = postc.center + axvar                             # Upper bounds per axis
    axlb = postc.center - axvar                             # Lower bounds per axis
    postc_all_pos = np.all( axlb >= 0 )
    
    
    # Get counterexamples from inclusion failure in the positive region
    bounds = (0, None)
    print("Checking positive side inclusion") #DEBUG
    check_lp_inclusion(ub_a, ub_b, eq_a, eq_b, prec.pos_m.T, prec.pos_b, bounds, n_cex, lcex)
        
    # Exit if enough cex found, or if postc is entirely positive
    if postc_all_pos or len(lcex) >= n_cex:
        print("Postcondition is entirely in positive domain") #DEBUG
        return lcex
           
    # If prec does not cover any negative side, then look for negative side points in postc
    if prec.neg_side_type == NegSideType.NONE:
        print("No negative side behavior for postcondition") #DEBUG
        for i in np.where(axlb < 0)[0]:
        
            # Build alpha and add cex
            alpha = np.ones(postc.reg_dim)
            alpha[ np.where(postc.basis[:,i] > 0) ] = -1
            print( "Positive side violation" )#DEBUG
            lcex.append( alpha @ postc.basis + postc.center )
            assert np.any(lcex[-1] < 0) #TODO remove DEBUG(?)
            if len(lcex) >= n_cex:
                return lcex
        
        return lcex
    
    # Else, add the plane seperating the positive and negative side to postc constraints
    nub_a = np.zeros(( ub_a.shape[0]+1, ub_a.shape[1] ))
    nub_b = np.zeros( ub_b.shape[0] + 1 )
    nub_a[:-1, :] = ub_a
    nub_b[:-1] = ub_b
    if prec.neg_side_type == NegSideType.QUAD:
        nub_a[ -1, prec.neg_side_quad ] = 1
        nub_b[-1] = 0
    else:
        for b, i in zip(prec.zer_b, prec.zer_i):
            nub_a[-1, i] = 1/b
        nub_b[-1] = 1
    
    # Get the negative side constriants
    ns_m, ns_b = prec.get_neg_constrs()
    
    # Get the cexes
    print("Looking for negative side cexes") #DEBUG
    check_lp_inclusion(nub_a, nub_b, eq_a, eq_b, ns_m.T, ns_b, (None, None), n_cex, lcex)
        
    return lcex


if __name__ == "__main__":
    
    #ia = np.array([ [1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]])
    #ib = np.array([1, 1, 1])
    #
    #oa = np.array([ [1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]])
    #ob = np.array([2, 2, 2])
    
    #ia = np.array([ [ 1,  0,  0],
    #                [ 0,  1,  0],
    #                [ 0,  0,  1],
    #                [-1,  0,  0],
    #                [ 0, -1,  0],
    #                [ 0,  0, -1]])
    #ib = np.array([1, 1, 1, 1, 1, 1])
    #
    #oa = np.array([ [ 1,  1,  1],
    #                [ 1,  1, -1],
    #                [ 1, -1,  1],
    #                [ 1, -1, -1],
    #                [-1,  1,  1],
    #                [-1,  1, -1],
    #                [-1, -1,  1],
    #                [-1, -1, -1]])
    #ob = np.array([2, 2, 2, 2, 2, 2, 2, 2])
    #
    #cex = []
    #check_lp_inclusion(ia, ib, None, None, oa, ob, (0, None), 100, cex)
    #print(cex)
    
    #prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), neg_side_type = NegSideType.ZERO)
    #postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([0.9, 0, 0]))
    #prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
    #                                        neg_side_type = NegSideType.QUAD,
    #                                        neg_side_quad = [1, 2])
    #postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.9, -0.9, -0.9]))
    #print(check_inclusion(postc, prec, n_cex = 100))
    
    from timeit import timeit
    import random
    import sys
    
    from scipy.stats import ortho_group
    
    from debug import rand_sparce_matrix, rand_sparce_pos_matrix
    
    
    n = 150         # Number of neurons
    k1 = 100         # Number of postcond bases
    k2 = 80         # Number of precond constriants
    cenvar = 10     # Range of values for postcond center
    basvar = 10     # Range of basis vector lengths
    bndvar = 10     # Range of values of precond bounds
    p0 = random.random() * 0.1      # Sparcity of postcond basis
    p1 = random.random() * 0.15     # Sparcity of precondn bounds
    n_run = 100
    n_cex = 5
    
    t = 0

    pre_m, pre_b, pst_v, pst_c, nsq, prec, postc = None, None, None, None, None, None, None
    nst = None
    
    def fail_dump():
        global pre_m, pst_v, pre_b, pst_c, nst, nsq
        # Dump basis, center and tie classes found to log if methods do not match.
        data = {}
        data['pre_m'] = pre_m.tolist()
        data['pre_b'] = pre_b.tolist()
        data['pst_v'] = pst_v.tolist()
        data['pst_c'] = pst_c.tolist()
        data['nst'] = nst
        data['nsq'] = nsq.tolist() if nsq is not None else nsq
        with open(sys.argv[1], 'w') as log:
            log.write(str(data))
        
    def run_pb():
        global prec, postc
        print("Running pullback")
        ret = check_inclusion(postc, prec, n_cex=n_cex)
    
    if len(sys.argv) >= 3 and sys.argv[2] == "checklog":
        with open(sys.argv[1]) as log:
            data = eval(log.read())
            pre_m = np.array(data['pre_m'])
            pre_b = np.array(data['pre_b'])
            pst_v = np.array(data['pst_v'])
            pst_c = np.array(data['pst_c'])
            nst = data['nst']
            nsq = np.array(data['nsq'])
            
            nst = random.random()
            if nst <= 0.333:
                prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.NONE)
            elif nst > 0.333 and nst <= 0.666 and np.all( pre_b >= 0 ):
                prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.ZERO)
            else:
                prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.QUAD, 
                                        neg_side_quad = np.where( nsq < 0.5) )
                
            postc = LinearPostcond(pst_v, pst_c)
            
            print("Running inclusion check")
            try:
                t += timeit(run_pb, number=1)
            except Exception as e:
                fail_dump()
                raise e
            
            exit()
            
    
    for i in range(n_run):
        print(f"Run {i} of {n_run}")

        print("Generating data")
        pre_m = rand_sparce_matrix(n,k1,p1)
        pst_v = ortho_group.rvs(n)[:k2, :] * basvar
        pre_b = (np.random.randn(k1) - 0.5) * bndvar
        pst_c = (rand_sparce_matrix(1,n,p0)[0,:] - 0.5) * cenvar
        
        nst = random.random()
        if nst <= 0.333:
            prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.NONE)
            print("NONE postcond")
        elif nst > 0.333 and nst <= 0.666 and np.all( pre_b >= 0 ):
            prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.ZERO)
            print("ZERO postcond")
        else:
            nsq = np.random.randn(n)
            prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.QUAD, 
                                    neg_side_quad = np.where( nsq < 0.5) )
            print("QUAD postcond")
            
        postc = LinearPostcond(pst_v, pst_c)
        
        print("Running inclusion check")
        try:
            t += timeit(run_pb, number=1)
        except Exception as e:
            fail_dump()
            raise e

    t /= n_run
    print(f"The average time for inclusion check is {t}")
