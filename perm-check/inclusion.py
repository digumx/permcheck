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
from global_consts import FLOAT_RTOL, FLOAT_ATOL, SCIPY_LINPROG_METHOD
from concurrency import log


MbFloat = Union[ None, float ]
MbArrayLike = Union[ ArrayLike, None ]


def check_lp_inclusion( inner_ub_a: MbArrayLike, inner_ub_b: MbArrayLike, 
                        inner_eq_a: MbArrayLike, inner_eq_b: MbArrayLike, 
                        outer_ub_a: MbArrayLike, outer_ub_b: MbArrayLike, 
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
        log(f"Checked constraint {i} of {len(outer_ub_a)}, found {len(l_cex)} cexes")
        i += 1
        
        res = linprog(-c, inner_ub_a, inner_ub_b, inner_eq_a, inner_eq_b, bounds = inner_bounds,
                method = SCIPY_LINPROG_METHOD)      # Call optimizer
        
        if res.status == 2:    # Inner is infeasable
            log("Infeasable inner") #DEBUG
            return
        
        elif res.status == 0:
            if -res.fun > b:    # Out of bounds, found cex
                l_cex.append(res.x)
                if len(l_cex) >= n_cex:
                    break
        
        else:
            raise RuntimeError("Optimizer returned bad status: {0}".format(res.status))
    
    log(f"Lp check leaves with {len(l_cex)} cexes")


    
def _pp_lcex(lcex, postc):     # Post processign for lcex before return TODO Remove duplicates
    if len(lcex) > 0:
        return np.concatenate( [ c[np.newaxis, :] for c in lcex ], axis=0) @ postc.basis +\
                        postc.center
    else:
        return []


def check_inclusion_pre_relu( postc: LinearPostcond, prec: DisLinearPrecond, n_cex : int = 1) \
                    -> list[ArrayLike]:
    """
    Check if given postcondition `postc` is included within the precondition `prec`, both being at a
    position just before a relu layer and just after a linear layer. Returns a list of upto `n_cex`
    counterxamples, or an empty list if inclusion is satisified.  Note returned list may contain
    duplicate elements.
    """
    assert postc.num_neuron == prec.num_neuron
   
    lcex = []
    
    # Check if postcondition is entirely within the positive region, or entirely negative.
    axvar = np.sum( np.absolute( postc.basis ), axis = 0 )  # Ub - Lb per axis
    axub = postc.center + axvar                             # Upper bounds per axis
    axlb = postc.center - axvar                             # Lower bounds per axis
    postc_all_pos = np.all( axlb >= 0 )
    
    # Lift the positive region to alpha
    p_reg_m = np.transpose(postc.basis @ prec.pos_m)
    p_reg_b = prec.pos_b - postc.center @ prec.pos_m
    
    # If postc is all positive, just check inclusion in the positive region
    if postc_all_pos or len(lcex) >= n_cex:
        log("Postcondition is entirely in positive domain") #DEBUG
        check_lp_inclusion(None, None, None, None, p_reg_m, p_reg_b, (-1, 1), n_cex-len(lcex), lcex)
        return _pp_lcex(lcex, postc)
    
    # Lift the positivity conditions for each axis to alpha space.
    pos_per_ax_m, pos_per_ax_b = -np.transpose(postc.basis), np.copy(postc.center)
    
    # If prec does not cover any negative side, then 
    if prec.neg_side_type == NegSideType.NONE:
        log("No negative side behavior for postcondition") #DEBUG
        
        # look for negative side points in postc
        for i in np.where(axlb < 0)[0]:
        
            # Build alpha and add cex
            alpha = np.ones(postc.reg_dim)
            alpha[ np.where(postc.basis[:,i] > 0) ] = -1
            log( "Positive side violation" )#DEBUG
            lcex.append(alpha)
            assert np.any(lcex[-1] < 0) #TODO remove DEBUG(?)
            if len(lcex) >= n_cex:
                return _pp_lcex(lcex, postc)
        
        # Look for points in positive quadrant not in positive region
        check_lp_inclusion(pos_per_ax_m, pos_per_ax_b, None, None, p_reg_m, p_reg_b, (-1, 1),
                                n_cex-len(lcex), lcex)
        return _pp_lcex(lcex, postcn)
    
    # Otherwise, get full LP for positive and negative region with quadrant constraints, and lift
    m, b = prec.get_pos_constrs()
    p_lp_m = np.transpose(postc.basis @ m)
    p_lp_b = b - postc.center @ m
    
    # And for negative region
    m, b = prec.get_neg_constrs()
    n_lp_m = np.transpose(postc.basis @ m)
    n_lp_b = b - postc.center @ m

    
    if prec.neg_side_type == NegSideType.QUAD:
        log("Quad inclusion check") #DEBUG
        
        # Find the plane seperating the positive and negative side
        sep_plane = np.zeros(prec.num_neuron)
        sep_plane[ prec.neg_side_quad ] = 1
        
        # Lift the sep plane to alpha side.
        spp_m = postc.basis @ sep_plane
        spp_b = - np.inner(postc.center, sep_plane)

        # Find cexes from above sep plane
        check_lp_inclusion(-spp_m[np.newaxis,:], np.array([-spp_b]), None, None, p_lp_m, p_lp_b, 
                                (-1, 1), n_cex-len(lcex), lcex)
        log(f"{len(lcex)} cexes after above sep plane") 
        if len(lcex) >= n_cex:
            return _pp_lcex(lcex, postc)
        
        # Find cexes from below sep plane
        check_lp_inclusion(spp_m[np.newaxis,:], np.array([spp_b]), None, None, n_lp_m, n_lp_b, 
                                (-1, 1), n_cex-len(lcex), lcex)
        log(f"{len(lcex)} cexes after below sep plane") 
        if len(lcex) >= n_cex:
            return _pp_lcex(lcex, postc)
        
    
    elif prec.neg_side_type == NegSideType.ZERO:
        
        # Look for points in positive quadrant not in positive region
        check_lp_inclusion(pos_per_ax_m, pos_per_ax_b, None, None, p_reg_m, p_reg_b, (-1, 1),
                                n_cex-len(lcex), lcex)
        if len(lcex) >= n_cex:
            return _pp_lcex(lcex, postc)
        
        # For each axis, check if there are points with negative value along the axis outside bounds
        for ax_m, ax_b in zip(pos_per_ax_m, pos_per_ax_b):
            check_lp_inclusion(-ax_m[np.newaxis,:], -ax_b, None, None, n_lp_m, n_lp_b, (-1, 1),
                                    n_cex-len(lcex), lcex)
            if len(lcex) >= n_cex:
                return _pp_lcex(lcex, postc)
        
        
    return _pp_lcex(lcex, postc)


def check_inclusion_pre_linear(postc : LinearPostcond, prec_m : ArrayLike, prec_b : ArrayLike,
                                n_cex : int = 1) -> list[ ArrayLike ] :
    """
    Checks if the given postcondition positioned after the relu is contained in the precondition
    given by the LP x @ prec_m <= prec_b.
    """
    # Lift the LP region to alpha
    p_reg_m = np.transpose(postc.basis @ prec_m)
    p_reg_b = prec_b - postc.center @ prec_m
    
    # Lift the positivity conditions for each axis to alpha space.
    pos_per_ax_m, pos_per_ax_b = -np.transpose(postc.basis), np.copy(postc.center)
    
    # Do the LP inclusion check
    lcex = []
    check_lp_inclusion(pos_per_ax_m, pos_per_ax_b, None, None, p_reg_m, p_reg_b, (-1, 1), n_cex,
            lcex)
    
    # Return
    log("\n\n ATTENTION : Returning {0} cex \n\n".format(len(lcex))) #DEBUG
    return _pp_lcex(lcex, postc)
    


if __name__ == "__main__":
    
    from timeit import timeit
    import random
    import sys
    
    from scipy.stats import ortho_group
    
    from debug import rand_sparce_matrix, rand_sparce_pos_matrix
        
    if sys.argv[1] == "unit":
    
        #prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), neg_side_type = NegSideType.ZERO)
        #postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([0.9, 0, 0]))
        #prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
        #                                        neg_side_type = NegSideType.QUAD,
        #                                        neg_side_quad = [1, 2])
        #postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.9, -0.9, -0.9]))
        #log(check_inclusion(postc, prec, n_cex = 100))
       
        
        if sys.argv[2] == "1":      # Should return nothing
            prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), neg_side_type = NegSideType.NONE)
            postc = LinearPostcond(np.array([[0, 0.1, 0.1], [0.1, 0, 0]]), np.array([0.5, 0.5, 0.5]))
            log(check_inclusion(postc, prec, n_cex = 100))
        
        elif sys.argv[2] == "2":    # Should produce a cex from positive side failure
            prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), 
                                                            neg_side_type = NegSideType.NONE)
            postc = LinearPostcond(np.array([[0, 0.1, 0.1], [0.6, 0, 0]]), np.array([0.5, 0.5, 0.5]))
            log(check_inclusion(postc, prec, n_cex = 100))
        
        elif sys.argv[2] == "3":    # Same as before, but should produce more cexes from negative side
            prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), 
                                                            neg_side_type = NegSideType.NONE)
            postc = LinearPostcond(np.array([[0, 1, 1], [0.6, 0, 0]]), np.array([0.5, 0.5, 0.5]))
            log(check_inclusion(postc, prec, n_cex = 100))
        
        elif sys.argv[2] == "4":    # Zero pullback should cover things around zero, no cexes
            prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), 
                                                            neg_side_type = NegSideType.ZERO)
            postc = LinearPostcond(np.array([[0, 0.5, 0.5], [0.5, 0, 0]]), np.array([0, 0, 0]))
            log(check_inclusion(postc, prec, n_cex = 100))
        
        elif sys.argv[2] == "5":    # Should violate the negative side 
            prec = DisLinearPrecond(np.array([[1, 1, 1]]).T, np.array([3]), 
                                                            neg_side_type = NegSideType.ZERO)
            postc = LinearPostcond(np.array([[0, 0.1, 0.1], [1.1, 0, 0]]), np.array([0, 0, 0]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
        elif sys.argv[2] == "6":    # Should jut out of positive region
            prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                                    neg_side_type = NegSideType.QUAD,
                                                    neg_side_quad = [1, 2])
            postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.9, -0.9, -0.9]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
        elif sys.argv[2] == "7":    # Should be sat
            prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                                    neg_side_type = NegSideType.QUAD,
                                                    neg_side_quad = [1, 2])
            postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.6, -0.9, -0.9]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
        elif sys.argv[2] == "8":    # Should be sat
            prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                                    neg_side_type = NegSideType.QUAD,
                                                    neg_side_quad = [1, 2])
            postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.9, -1, -1]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
        elif sys.argv[2] == "9":    # Should be sat
            prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                                    neg_side_type = NegSideType.QUAD,
                                                    neg_side_quad = [1, 2])
            postc = LinearPostcond(np.array([[0, 1, 1], [0.1, 0, 0]]), np.array([2.15, -0.7, -0.7]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
        elif sys.argv[2] == "10":    # Should be sat
            prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                                    neg_side_type = NegSideType.QUAD,
                                                    neg_side_quad = [1, 2])
            postc = LinearPostcond(np.array([[0, -0.4, 0.4], [-0.4, 0.4, 0]]), 
                                                    np.array([0.8, 0.8, 0.8]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
        
        elif sys.argv[2] == "11":    # Jut out of negative region
            prec = DisLinearPrecond(np.array([[1, 1, 1], [-1, -1, -1]]).T, np.array([3, -2]), 
                                                    neg_side_type = NegSideType.QUAD,
                                                    neg_side_quad = [1, 2])
            postc = LinearPostcond(np.array([[0, -0.5, 0.5], [-0.5, 0.5, 0]]), 
                                                np.array([1.1, 1.1, 0.3]))
            log(check_inclusion(postc, prec, n_cex = 100))
    
    
    elif sys.argv[1] == "fuzz":
        
        n = 150         # Number of neurons
        k1 = 100        # Number of postcond bases
        k2 = 80         # Number of precond constriants
        cenvar = 10     # Range of values for postcond center
        basvar = 10     # Range of basis vector lengths
        bndvar = 10     # Range of values of precond bounds
        p0 = random.random() * 0.1      # Sparcity of postcond basis
        p1 = random.random() * 0.15     # Sparcity of precondn bounds
        n_run = 100
        n_cex = 40
        
        t = 0

        pre_m, pre_b, pst_v, pst_c, nsq, prec, postc = None, None, None, None, None, None, None
        nst = None
        n_z = 0
        n_q = 0
        n_n = 0
        
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
            with open(sys.argv[2], 'w') as log:
                log.write(str(data))
            
        def run_pb():
            global prec, postc
            log("Running pullback")
            ret = check_inclusion(postc, prec, n_cex=n_cex)
        
        if len(sys.argv) >= 4 and sys.argv[3] == "checklog":
            with open(sys.argv[2]) as log:
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
                
                log("Running inclusion check")
                try:
                    t += timeit(run_pb, number=1)
                except Exception as e:
                    fail_dump()
                    raise e
                
                exit()
                
        
        for i in range(n_run):
            log(f"Run {i} of {n_run}")

            log("Generating data")
            pre_m = rand_sparce_matrix(n,k2,p1)
            pst_v = ortho_group.rvs(n)[:k1, :] * basvar
            pre_b = (np.random.rand(k2) - 0.5) * bndvar
            pst_c = (rand_sparce_matrix(1,n,p0)[0,:] - 0.5) * cenvar
            
            nst = random.random()
            if nst <= 0.333:
                prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.NONE)
                n_n += 1
                log("NONE postcond")
            elif nst > 0.333 and nst <= 0.666:
                pre_b = np.random.rand(k2) * 0.5 * bndvar  # pre_b >= 0
                prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.ZERO)
                n_z += 1
                log("ZERO postcond")
            else:
                nsq = np.random.randn(n)
                prec = DisLinearPrecond(pre_m, pre_b, neg_side_type = NegSideType.QUAD, 
                                        neg_side_quad = np.where( nsq < 0.5) )
                n_q += 1
                log("QUAD postcond")
                
            postc = LinearPostcond(pst_v, pst_c)
            
            log("Running inclusion check")
            try:
                t += timeit(run_pb, number=1)
            except Exception as e:
                fail_dump()
                raise e

        t /= n_run
        log(f"The average time for inclusion check is {t}")
        log(f"There were {n_n} NONE, {n_z} ZERO and {n_q} QUAD")
