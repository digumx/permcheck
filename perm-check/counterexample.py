"""
Contains methods for pulling back counterexamples over neural network layers, and pushing them
forward.
"""


from typing import Union
from math import exp

import numpy as np
from numpy.random import rand
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.linalg import lstsq, null_space

from postcondition import LinearPostcond
from global_consts import CEX_PULLBACK_NUM_PULLBACKS, CEX_PULLBACK_SAMPLES_SCALE
from global_consts import CEX_PULLBACK_MAX_SAMPLES
from global_consts import FLOAT_RTOL, FLOAT_ATOL, SCIPY_LSTSQ_METHOD, NUMPY_SORT_METHOD
from global_consts import SCIPY_LINPROG_METHOD
from concurrency import log



def pullback_cex(   cex : ArrayLike, postc : LinearPostcond, 
                    weights : ArrayLike, bias : ArrayLike, 
                    n_pullbacks : int = CEX_PULLBACK_NUM_PULLBACKS,
                    max_samples : int = CEX_PULLBACK_MAX_SAMPLES,
                    samples_scale : int = CEX_PULLBACK_SAMPLES_SCALE ) -> list[ArrayLike] :
    """
    Pulls a cex back over layer to a LinearPostcond. Returns a list of points which may or may not
    be an actual pullback. The given cex point gives a set of inputs to the relu of the current
    layer, or a set of outputs from the previous layer. Each point gives a set of values at the
    input to the relu of the previous layer. This picks several samples from the space of alpha
    values uniformly at random, scores them according to their distance from y, and returns the top
    scoring ones after projecting them to the inverse image of the cex point as far as possible.
    Note that the returned points may not go to given cex across the layer, but do satisfy the given
    postcondition. Arguments are:
    
    cex             -   The counterexample point to pullback.
    postc           -   The postcondition to which to pull the counterexample back. If a pullback is
                        returned, it will be from this postcondition.
    weights,        -   The weights and bias over the DNN layer over which to pull back.
    bias
    n_pullbacks     -   The number of pullbacks to return.
    max_samples     -   The maximum number of samples to draw.
    samples_scale   -   A mutliplier for the number of random samples to perform. The actual number
                        is this times the sum of the dimensions of the involved layers.
                    
    Returns a list of pullbacks.
    """
    # Sanity checks TODO remove
    assert cex.ndim == 1
    assert weights.ndim == 2
    assert bias.ndim == 1
    assert postc.num_neuron == weights.shape[0]
    assert cex.shape[0] == weights.shape[1]
    assert bias.shape[0] == weights.shape[1]
    
    n = sum(weights.shape)
    
    # Get the samples
    log("Sampling {0} samples".format(samples_scale * n))
    n_samps = min(samples_scale * n, max_samples)
    samples = rand(samples_scale * n , postc.reg_dim)
    
    # Score the samples
    log("Scoring {0} samples".format(samples.shape[0]))
    x_vals = samples @ postc.basis + postc.center
    x_vals[ np.where(x_vals < 0)[0] ] = 0
    disps = x_vals @ weights + bias - cex
    score = np.einsum("...i,...i->...", disps, disps)
    
    # Pick the best.
    bst_idx = np.argsort(score, kind=NUMPY_SORT_METHOD)[:n_pullbacks]
    cex_cands = samples[ bst_idx, : ]
    
    # Project them, clamp to alpha bounds
    ret = []
    for i, cand in enumerate(cex_cands):
        
        log("Projecting {0} of {1} candidates".format(i, len(cex_cands)))

        # Get linear map
        x_val = cand @ postc.basis + postc.center
        quad = np.ones((1, postc.center.shape[0]))
        quad[ 0, np.where( x_val < 0 )[0] ] = 0
        A = (quad * postc.basis) @ weights
        b = cex - (quad[0, :] * postc.center) @ weights - bias
        
        # Solve least squares to get a point
        p, _, _, _ = lstsq(A.T, b, cond = FLOAT_ATOL, lapack_driver = SCIPY_LSTSQ_METHOD)
        
        # Project candidate and clamp to alpha
        n_A = null_space(A.T, rcond = FLOAT_ATOL)
        #log(f"A: {A.shape}, n_A: {n_A.shape}, p: {p.shape}, cand: {cand.shape}") #DEBUG
        if n_A.shape[0] >= 1: 
            proj_p = p + (cand - p) @ n_A @ n_A.T
        else:
            proj_p = cand
        proj_p[ np.where( proj_p >  1 )] =  1
        proj_p[ np.where( proj_p < -1 )] = -1
        
        ret.append(proj_p @ postc.basis + postc.center)
    
    return ret


def pullback_cex_linear( cex : ArrayLike, postc : LinearPostcond, weights : ArrayLike, 
                        bias : ArrayLike ) -> Union[ ArrayLike, None ] :
    """
    Pulls back a given counterexample over a linear layer using a linear program call. Returns a CEX
    point if found, else returns none
    """
    # TODO remove
    assert cex.ndim == 1
    assert weights.ndim == 2
    assert bias.ndim == 1
    assert bias.shape[0] == weights.shape[1]
    assert cex.shape[0] == weights.shape[1]
    assert postc.num_neuron == weights.shape[0]
    
    # Run linear program
    res = linprog(c = np.zeros((postc.reg_dim)), A_ub = None, b_ub = None, 
                    A_eq = np.transpose( postc.basis @ weights ), 
                    b_eq = cex - bias - postc.center @ weights,
                    bounds = (-1, 1),
                    method = SCIPY_LINPROG_METHOD )
    
    # Analyze result
    if res.status == 0:
        log("Pullback across linear layer successful")
        return res.x @ postc.basis + postc.center

    elif res.status == 1 or res.status == 2:
        log("Pullback failed, preimage of point does not intersect postcondition")
        return None
    
    else:
        raise RuntimeError("Unexpected return status from linear solver: {0}".format(res.status))

    
def pullback_cex_relu(cex : ArrayLike, postc : LinearPostcond) -> Union[ ArrayLike, None ]:
    """
    Pulls a counterexample back over a ReLU layer into a given linear postcondition. Returns a point
    if it is found, else returns None.
    """
    assert cex.ndim == 1
    assert postc.num_neuron == cex.shape[0]
    
    # Early exit if not in positive quadrant
    if np.any( cex < -FLOAT_ATOL ):
        log("Cex not in positive quadrant, cannot pull back over ReLU")
        return None
    
    # Get LP for inverse image of point under ReLU and lift it to alpha space
    proj = np.eye(postc.num_neuron)[ :, np.where(cex > FLOAT_ATOL)[0] ]
    A_eq = np.transpose( postc.basis @ proj)
    b_eq = (cex - postc.center) @ proj
    
    # Get quadrant bounds for inverse image
    if np.all( cex > FLOAT_ATOL):       # No bounds
        A_ub = b_ub = None
    else:
        pos = np.eye(postc.num_neuron)[ :, np.where(cex <= FLOAT_ATOL)[0] ]
        A_ub = np.transpose( postc.basis @ pos )
        b_ub = postc.center @ pos
    
    # Run linear program
    res = linprog(np.zeros((A_ub.shape[1])), A_ub, b_ub, A_eq, b_eq,
                    bounds = (-1, 1),
                    method = SCIPY_LINPROG_METHOD )
    
    # Analyze result
    if res.status == 0:
        log("Pullback across relu layer successful")
        return res.x @ postc.basis + postc.center

    elif res.status == 1 or res.status == 2:
        log("Pullback failed, preimage of point does not intersect postcondition")
        return None
    
    else:
        raise RuntimeError("Unexpected return status from linear solver: {0}".format(res.status))
    


if __name__ == "__main__":
    
    from timeit import timeit
    import random
    import sys
    
    from scipy.stats import ortho_group
    
    from debug import rand_sparce_matrix, rand_sparce_pos_matrix
    
    def check_pb(cex, pbs, basis, center, weights, bias):
        """
        Retur average and minimum score for pullbacks.
        """
        x_vals = [ pb @ basis + center for pb in pbs ]
        for xv in x_vals: xv[ np.where( xv < 0 )] = 0
        scores = [ np.linalg.norm( xv @ weights + bias - cex ) for xv in x_vals ]
        return sum(scores) / len(scores), min(scores)
        
    
    if sys.argv[1] == "unit":
        
        #basis =     np.array([[0.1, 1, 1], [1, 0.1, 0]])
        #center =    np.array([2, 1, 0.5])
        
        if sys.argv[2] == "0":
            basis =     np.array([[1, 0, 0], [0, 1, 0]])
            center =    np.array([0, 0, 0])
            weights =   np.array([[1, 0], [0, 1], [0, 0]])
            bias =      np.array([0, 0])
            alph =      np.array([0.2, 0.4])
            init_alph = np.array([0.5, 0.5])
            x = alph @ basis + center
            x[ np.where( x < 0 )] = 0
            y = x @ weights + bias
            log(y)
            log(x)
            pbs = pullback_cex(y, LinearPostcond(basis, center), weights, bias)
            log(pbs)
            a, m = check_pb(y, pbs, basis, center, weights, bias)
            log(f"Average score {a}, minimum score {m}")
        
        if sys.argv[2] == "1":
            basis =     np.array([[1, 0, 0], [0, 1, 0]])
            center =    np.array([0, 0, 0])
            weights =   np.array([[1, 2], [0.1, 2], [5, 0.5]])
            bias =      np.array([0, 1])
            alph =      np.array([0.2, 0.4])
            init_alph = np.array([0.5, 0.5])
            x = alph @ basis + center
            x[ np.where( x < 0 )] = 0
            y = x @ weights + bias
            log(y)
            log(x)
            pbs = pullback_cex(y, LinearPostcond(basis, center), weights, bias)
            log(pbs)
            a, m = check_pb(y, pbs, basis, center, weights, bias)
            log(f"Average score {a}, minimum score {m}")
        
        elif sys.argv[2] == "2":
            basis =     np.array([[1, 0, 0], [0, 1, 0]])
            center =    np.array([0, 0, 0])
            weights =   np.array([[1, 2], [0.1, 2], [5, 0.5]])
            bias =      np.array([0, 1])
            alph =      np.array([0.2, 0.4])
            init_alph = np.array([0.5, -0.5])
            x = alph @ basis + center
            x[ np.where( x < 0 )] = 0
            y = x @ weights + bias
            log(y)
            log(x)
            pbs = pullback_cex(y, LinearPostcond(basis, center), weights, bias)
            log(pbs)
            a, m = check_pb(y, pbs, basis, center, weights, bias)
            log(f"Average score {a}, minimum score {m}")
        
        elif sys.argv[2] == "3":
            basis =     np.array([[1, 0, 0], [0, 1, 0]])
            center =    np.array([0, 0, 0])
            weights =   np.array([[1, 2], [0.1, 2], [5, 0.5]])
            bias =      np.array([0, 1])
            alph =      np.array([-0.2, 0.4])
            init_alph = np.array([0.5, -0.5])
            x = alph @ basis + center
            x[ np.where( x < 0 )] = 0
            y = x @ weights + bias
            log(y)
            log(x)
            pbs = pullback_cex(y, LinearPostcond(basis, center), weights, bias)
            log(pbs)
            a, m = check_pb(y, pbs, basis, center, weights, bias)
            log(f"Average score {a}, minimum score {m}")
        
        elif sys.argv[2] == "4":
            basis =     np.array([[1, 0, 0], [0, 1, 0]])
            center =    np.array([0, 0, 0])
            weights =   np.array([[1, 2], [0.1, 2], [5, 0.5]])
            bias =      np.array([0, 1])
            alph =      np.array([0.2, 0.4])
            x = alph @ basis + center
            x[ np.where( x < 0 )] = 0
            y = x @ weights + bias
            log(y)
            log(x)
            pbs = pullback_cex(y, LinearPostcond(basis, center), weights, bias)
            log(pbs)
            a, m = check_pb(y, pbs, basis, center, weights, bias)
            log(f"Average score {a}, minimum score {m}")
        
        elif sys.argv[2] == "5":
            basis =     np.array([[1, 0, 0], [0, 1, 0]])
            center =    np.array([0, 0, 0])
            weights =   np.array([[1, 2], [0.1, 2], [5, 0.5]])
            bias =      np.array([0, 1])
            alph =      np.array([-0.2, 0.4])
            x = alph @ basis + center
            x[ np.where( x < 0 )] = 0
            y = x @ weights + bias
            log(y)
            log(x)
            pbs = pullback_cex(y, LinearPostcond(basis, center), weights, bias)
            log(pbs)
            a, m = check_pb(y, pbs, basis, center, weights, bias)
            log(f"Average score {a}, minimum score {m}")
            
    elif sys.argv[1] == "fuzz":

        n1 = 50                         # Number of neurons in current layer
        n2 = 80                         # Number of neurons in next layer  
        k = 40                          # Number of precond constriants
        cenvar = 10                     # Range of values for postcond center
        basvar = 10                     # Range of basis vector lengths
        weivar = 10                     # Range of values of weights
        biavar = 10                     # Range of values of bias
        pw = random.random() * 0.20     # Sparcity of weights
        
        n_run = 1000
        
        t = 0

        weights = bias = basis = center = cex_alph = postc = cex = ret = None  
        n_sat = 0
        av, mn = 0, None
        
        def fail_dump():
            global weights, bias, center, basis, cex_alph, postc, cex
            # Dump basis, center and tie classes found to log if methods do not match.
            data = {}
            data['weights']     = weights.tolist()
            data['bias']        = bias.tolist()
            data['basis']       = basis.tolist()
            data['center']      = center.tolist()
            data['cex_alph']    = cex_alph.tolist()
            with open(sys.argv[2], 'w') as log:
                log.write(str(data))
            
        def run_pb():
            global postc, cex, weights, bias, ret, n_reps
            ret = pullback_cex(cex, postc, weights, bias)
        
        if len(sys.argv) >= 4 and sys.argv[3] == "checklog":
            with open(sys.argv[2]) as log:
                data = eval(log.read())
                weights     = np.array(data['weights'])
                bias        = np.array(data['bias'])
                basis       = np.array(data['basis'])
                center      = np.array(data['center'])
                cex_alph    = np.array(data['cex_alph'])
                
                cx = cex_alph @ basis + center
                cx[ np.where( cx < 0 ) ] = 0
                cex = cx @ weights + bias
                postc = LinearPostcond(basis, center)
                
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
            weights     = rand_sparce_matrix(n1, n2, pw) * weivar
            bias        = (np.random.rand(n2) - 0.5) * biavar
            basis       = ortho_group.rvs(n1)[:k, :] * basvar
            center      = (np.random.rand(n1) - 0.5) * cenvar
            cex_alph    = (np.random.rand(k) - 0.5) * 2
            
            cx = cex_alph @ basis + center
            cx[ np.where( cx < 0 ) ] = 0
            cex = cx @ weights + bias
            postc = LinearPostcond(basis, center)
            
            log("Running inclusion check")
            try:
                t += timeit(run_pb, number=1)
            except Exception as e:
                fail_dump()
                raise e
            
            a, m = check_pb(cex, ret, basis, center, weights, bias)
            av += a
            if mn is not None:
                mn = min(mn, m)
            else:
                mn = m
            if m < FLOAT_ATOL:
                n_sat += 1
            

        t /= n_run
        log(f"The average time for inclusion check is {t}")
        log(f"There were {n_sat} of {n_run} successfull pullbacks")
        log(f"The average score was {av/n_run}, the minimum was {mn}")
