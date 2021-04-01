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


def check_lp_inclusion(inner_ub_a: ArrayLike, inner_ub_b: ArrayLike, inner_eq_a: ArrayLike,
                        inner_eq_b: ArrayLike, outer_ub_a: ArrayLike, outer_ub_b: ArrayLike, 
                        inner_bounds: Union[ list[ tuple[ float, float ] ], tuple[ float, float] ], 
                        n_cex: int, s_cex: set[ArrayLike]) -> None:
    """
    Checks weather the region { x : inner_ub_a @ x <= inner_ub_b, inner_eq_a @ x == inner_eq_b ,
    bounds[0] <= x <= bounds[1]} is contained within the region { x : outer_ub_a @ x <= outer_ub_b }
    via a number of linear program calls. Adds counterexamples found to `s_cex`, until the size of
    `s_cex` reaches `n_cex`, at which point it returns.
    """
    
    # Loop over all outer constraints, optimize for max value.
    for c, b in zip(outer_ub_a, outer_ub_b):
        res = linprog(-c, inner_ub_a, inner_ub_b, inner_eq_a, inner_eq_b, bounds = inner_bounds,
                method = SCIPY_LINPROG_METHOD)      # Call optimizer
        
        if res.status == 2:    # Inner is infeasable
            return []
        
        elif res.status == 0:
            if -res.fun > b[0]:    # Out of bounds, found cex
                s_cex.add(res.x)
                if len(lcex) >= n_cex:
                    break
        
        else:
            raise RuntimeError("Optimizer returned bad status: {0}".format(res.status))
        


def check_inclusion( postc: LinearPostcond, prec: DisLinearPrecond, n_cex : int = 1)
                    -> list[ArrayLike]:
    """
    Check if given postcondition `postc` is included within the precondition `prec`. Returns a list
    of upto `n_cex` counterxamples with no repetitions, or an empty list if inclusion is satisified.
    """
    assert postc.num_neuron == prec.num_neuron
    
    # Get the LP for the postcondition. This works only if the postcondition has orthogonal bases
    assert REDUCE_POSTCOND_BASIS    #TODO implement other case
    
    lcex = set()
    
    # Calculate the bounds.
    ub_a = np.concatenate( (postc.basis, -postc.basis), dim=0 )
    t1 = postc.basis @ postc.center
    t2 = np.einsum("...i,...i->...", postc.basis, postc.basis)
    ub_b = np.concatenate( (t2 + t1, t2 - t1), dim=0 )
    
    # Equality should say that all components perpendicular to basis should have same value as
    # center
    if postc.perp_basis is not None:
        eq_a = post.perp_basis
    else:
        pb = np.transpose( null_space(post.basis, rcond=FLOAT_RTOL) )
    eq_b = eq_a @ postc.center
        
    # Check if postcondition is entirely within the positive region, or entirely negative.
    axvar = np.sum( np.absolute( postc.basis ), dim = 0 )[0]    # Ub - Lb per axis
    axub = postc.center + axvar                                 # Upper bounds per axis
    axlb = postc.center - axvar                                 # Lower bounds per axis
    postc_all_pos = np.all( axlb >= 0 )
    postc_any_pos = np.any( axlb >= 0 )
    
    # Get counterexamples from inclusion failure in the positive region
    if postc_any_pos:
        bounds = (0, None)
        check_lp_inclusion(ub_a, ub_b, eq_a, eq_b, prec.pos_m, prec.pos_b, bounds, n_cex, lcex)
        
    # Exit if enough cex found, or if postc is entirely positive
    if postc_all_pos or len(lcex) >= n_cex:
        return list(lcex)
           
    # If prec does not cover any negative side, then look for negative side points in postc
    if prec.neg_side_type == NegSideType.NONE:
        for _, i in zip(np.where(axlb < 0)):
        
            # Build alpha and add cex
            alpha = np.ones(postc.reg_dim)
            alpha[ np.where(postc.basis[:,i] > 0) ] = -1
            lcex.add( alpha @ postc.basis + postc.center )
            if len(lcex) >= n_cex:
                return list(lcex)
        
        return list(lcex)
    
    # Else for QUAD, check if all postc points in the negative side belong to right quadrant
    elif prec.neg_side_type == NegSideType.QUAD:
    
        # A vector representing a line through both quads
        qperp = np.zeros(prec.num_neuron)               
        qperp[ prec.neg_side_quad ] = 1 
        
        # Use linearity to check if postc has points outside iehter quadrant, add those as cex
        if postc_any_pos:
            pll = check_parallel(postc.basis[:, prec.neg_side_quad], postc.center[prec.neg_side_quad])
            for i in np.where(not pll):
                alhpa =  - np.inner(postc.center, qperp) / np.inner(postc.basis[i, :], qperp)
                lcex.add( alpha * postc.basis[i, :] + postc.center )
                if len(lcex) > n_cex:
                    return list(lcex)
            
        # Otherwise, check inclusion within the negative side region
        bounds = [ ( (None, 0) if i in prec.neg_side_quad else (0, None) )
                                for i in range(postc.num_neuron) ]
        check_lp_inclusion(ub_a, ub_b, eq_a, eq_b, prec.neg_m, prec.neg_b, bounds, n_cex, lcex)
        if len(lcex) >= n_cex:
            return list(lcex)
        
    
    # Finally, deal with the case when the negative side is around zero.     
    elif prec.neg_side_type == NegSideType.ZERO:
        
        rem_bnds = []
        
        # For each bound on the negative side, see if we can easily prove boundedness or find cex by
        # checking if the bound is satisfied regardless of what side we are on
        for b, i in zip(prec.zer_b, prec.zer_i):
            
            # Bound satisfied, move on
            if axub[i] <= b:
                continue
                
            # Find extremal point.
            alpha = np.ones(postc.reg_dim)
            alpha[ np.where(postc.basis[:,i] < 0) ] = -1
            x = alpha @ postc.basis + postc.center
            
            # Check if point is all positive, if so add as cex.
            if np.any( x < 0 ):
                lcex.add(x)
                if len(lcex) >= n_cex:
                    return list(lcex)
                continue
            
            # Else, save for later
            rem_bnds.append((b, i))
        
        # Construct lp for remaining bounds
        blp_a = np.zeros(( len(rem_bnds), postc.num_neuron ))
        blp_b = np.zeros( len(rem_bnds) )
        for i, bnd in enumerate(rem_bnds):
            blp_a[i, bnd[1]] = 1
            blp_b[i] = bnd[0]
        
        # For each axis, use lp to add cexes
        for i in range(prec.num_neuron):
            bounds = [ (None, 0 if j == i else None) for j in range(prec.num_neuron) ]
            check_lp_inclusion(ub_a, ub_b, eq_a, eq_b, blp_a, blp_b, bounds, n_cex, lcex)
            if len(lcex) >= n_cex:
                return list(lcex)
            
    else:
        raise ValueError("Precondition has unknown negative side type {0}".format(
                                prec.neg_side_type))
        
    return list(lcex)
