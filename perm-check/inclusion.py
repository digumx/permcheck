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
    for c, b in zip(outer_ub_a, outer_ub_b):
        
        res = linprog(-c, inner_ub_a, inner_ub_b, inner_eq_a, inner_eq_b, bounds = inner_bounds,
                method = SCIPY_LINPROG_METHOD)      # Call optimizer
        
        if res.status == 2:    # Inner is infeasable
            return []
        
        elif res.status == 0:
            if -res.fun > b:    # Out of bounds, found cex
                l_cex.append(res.x)
                if len(l_cex) >= n_cex:
                    break
        
        else:
            raise RuntimeError("Optimizer returned bad status: {0}".format(res.status))
        


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
    postc_any_pos = np.any( axlb >= 0 )
    
    
    # Get counterexamples from inclusion failure in the positive region
    if postc_any_pos:
        bounds = (0, None)
        print(" Checking positive side inclusion")
        check_lp_inclusion(ub_a, ub_b, eq_a, eq_b, prec.pos_m.T, prec.pos_b, bounds, n_cex, lcex)
        print(lcex)     #DEBUG
        
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
            print(lcex)#DEBUG
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
    prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                            neg_side_type = NegSideType.QUAD,
                                            neg_side_quad = [1, 2])
    postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.9, -0.9, -0.9]))
    print(check_inclusion(postc, prec, n_cex = 100))
