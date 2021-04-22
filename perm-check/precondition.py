"""
Contains classes and functions defining and pulling back the precondition across a layer of the
neural network.
"""


from typing import Union
from enum import Enum, auto

import numpy as np
from numpy.typing import ArrayLike

from global_consts import FLOAT_ATOL
from concurrency import log


class NegSideType(Enum):
    """
    An enum with the allowed kinds of conditions in the negative side in a precondition
    """
    NONE    = auto()    # No negative side condition, precondition is entirely positive
    QUAD    = auto()    # Negative side behavior is obtained via a pullback around a quadrantl
    ZERO    = auto()    # Obtained via pulling back around 0.


class DisLinearPrecond:
    """
    Class representing a linear precondition given by a disjunction of 2 linear postconditions, one
    describing the positive side behavior, the other describing the negative side behavior.
    
    NOTE: Niether the positive side constraints nor the negative side constraints actually enforce
    the fact that they contain only points from a certain quadrant. These constraints are assumed to
    be implicit.
    
    Members:
    
    num_neuron : int            -   Number of neurons
    pos_m, pos_b : ArrayLike    -   Specifies the positive side behavior as the linear region
                                    x @ pos_m <= pos_b                             
    neg_side_type : NegSideType -   What kind of condition exists in the negative side. Depending on
                                    the value of this member, the other members are:
                                    
    
    if neg_side_type is NONE, the class has no further members
    
    
    if neg_side_type is QUAD, the class aditionally has:
    
    neg_side_quad : ArrayLike   -   An array containing the axes that are negative in the quadrant
                                    covering the negative side.
    neg_m, neg_b : ArrayLike    -   Specifies the negative side behavior as the linear region
                                    x @ neg_m <= neg_b
                                    
    
    if neg_side_type is ZERO, the class has:
    
    zer_b, zer_i : ArrayLike    -   The negative side behavior when taken around zero. For each
                                    `i`, the `zer_i[i]`-th axis is bounded by `zer_b[i]`.
    """
    
    def __init__(self, pos_m, pos_b, neg_side_type: NegSideType = NegSideType.NONE,
                    neg_side_quad : Union[None, ArrayLike] = None, 
                    neg_m : Union[None, ArrayLike] = None, neg_b : Union[None, ArrayLike] = None):
        """
        Construct a `DisLinearPrecond` with given positive side constraints, kind of negative side
        behavior to capture and (optionally) which negative side quadrant to cover. The actual
        constraints for the negative side is constructed automatically from the given data. The
        arguments are: 
        
        pos_m, pos_b    -   Constraints for the positive side behavior must have valid shapes.
        neg_side_type   -   Type of negaitive side behavior. Defaults to no negative side behavior.
                            Depending on it's value, some other arguments must be given.
        neg_side_quad   -   Must be given if `neg_side_type` is `QUAD`, in which case it is a list
                            of axes that are negative.
                            
        NOTE: pos_m and pos_b are not copied, but moved over.
        """
        
        # TODO remove shape asserts
        assert pos_m.ndim == 2
        assert pos_b.ndim == 1
        assert pos_m.shape[1] == pos_b.shape[0]
        
        # Set up basic stuff
        self.num_neuron : int = pos_m.shape[0]
        self.pos_m : ArrayLike = pos_m
        self.pos_b : ArrayLike = pos_b
        self.neg_side_type : NegSideType = neg_side_type
        
        # Do a quad construction
        if self.neg_side_type == NegSideType.QUAD:
            
            # Check if quad has been given
            if neg_side_quad is None:
                raise ValueError("Valid negative side quadrant must be provided")
            self.neg_side_quad : ArrayLike = neg_side_quad
            
            # Calculate negative side constraints. Given a particluar constraint, that can be
            # extended into any negative side quadrant by setting all the negative axes in the
            # quadrant to 0.
            self.neg_m : ArrayLike = np.copy(self.pos_m)
            self.neg_m[self.neg_side_quad, :] = 0
            self.neg_b : ArrayLike = np.copy(self.pos_b)
            
        # Calculate the bounds around 0
        elif self.neg_side_type == NegSideType.ZERO:
            
            # This must hold for ZERO to work, and is guaranteed if zero is in the region
            assert np.all(self.pos_b >= 0) # TODO use tolerance?
            
            mat = self.pos_m * self.pos_m.shape[0]
            mat[np.where(mat < FLOAT_ATOL)] = 0         # Zero and negative are set to 0
            mat = mat / self.pos_b                      # For each eqn j, get [ n * m_i / b_j ]
            bnds = np.amax(mat, axis=1)                 # Inverse of these are bounds
            self.zer_i : ArrayLike = np.where(bnds > 0)[0]  # zer_i is not a tuple, take first elt.
            self.zer_b : ArrayLike = 1 / bnds[self.zer_i]
            
        else:
            assert self.neg_side_type == NegSideType.NONE   # Do nothing
            
    
    def __repr__(self):
        """
        Return a representation that contains all information about this object
        """
        d = {}
        
        d['pos_m'] = self.pos_m.tolist()
        d['pos_b'] = self.pos_b.tolist()
        d['n'] = self.num_neuron
        
        if self.neg_side_type == NegSideType.NONE:
            d['neg_side_type'] = 'NONE'
        
        elif self.neg_side_type == NegSideType.ZERO:
            d['neg_side_type'] = 'ZERO'
            d['zer_i'] = self.zer_i.tolist()
            d['zer_b'] = self.zer_b.tolist()
        
        elif self.neg_side_type == NegSideType.QUAD:
            d['neg_side_type'] = 'QUAD'
            d['neg_m'] = self.neg_m.tolist()
            d['neg_b'] = self.neg_b.tolist()
            d['neg_side_quad'] = self.neg_side_quad.tolist()
        else:
            assert False
            
        return "DisLinearPrecond({0})".format(repr(d))
    
    def get_pos_constrs(self) -> tuple[ArrayLike, ArrayLike]:
        """
        Returns the complete set of constraints capturing the positive side behavior of the
        postcondition. Returns a pair `(m, b)`, so that the positive side region is given by
        {x : x * m <= b}. Note that these are not just `pos_m` and `pos_b`, as these do not enforce
        x to be within the all positive quadrant.
        """
        m = np.zeros((self.num_neuron, self.pos_m.shape[1] + self.num_neuron))
        b = np.zeros((self.pos_m.shape[1] + self.num_neuron))
        
        m[range(self.num_neuron), range(self.pos_m.shape[1], self.pos_m.shape[1] + self.num_neuron)] = -1  # Set positivity
        
        m[:,:self.pos_m.shape[1]] = self.pos_m                  # Original constraints
        b[:self.pos_m.shape[1]] = self.pos_b                    # Original constraints, rest 0
   
        return (m, b)
    
    def get_neg_constrs(self) -> Union[tuple[ArrayLike, ArrayLike], None]:
        """
        Returns the complete set of constraints capturing the negative side behavior of the
        postcondition. Returns a pair `(m, b)`, so that the negative side region is given by
        {x : x * m <= b}, or `None` if no negative side behavior is present. Note that the returned
        constraints enforce x to belong in the appropriate quadrant in the negative side. Note that
        returned constraints may be views into internal data, and modifying them may modify the
        precondition.
        """
        if self.neg_side_type == NegSideType.QUAD:
            
            m = np.zeros((self.num_neuron, self.neg_m.shape[1] + self.num_neuron))
            b = np.zeros((self.neg_m.shape[1] + self.num_neuron))
            
            s = -1 * np.ones(self.num_neuron)                                # Set up quadrant constraints
            s[self.neg_side_quad] = 1                               # And copy them
            m[range(self.num_neuron), range(self.neg_m.shape[1], self.neg_m.shape[1] + self.num_neuron)] = s
            
            m[:,:self.neg_m.shape[1]] = self.neg_m                  # Original constraints
            b[:self.neg_m.shape[1]] = self.neg_b                    # Original constraints, rest 0
            
            return m, b
        
        elif self.neg_side_type == NegSideType.ZERO:
            
            m = np.zeros(( self.num_neuron, self.zer_i.shape[0] ))
            m[ self.zer_i, range(self.zer_i.shape[0]) ] = 1         # Set up diagonals
            
            return m, self.zer_b
        
        else:
            
            return None
   
   
   
def pull_back_constr_relu(ms: list[ArrayLike], bs: list[ArrayLike], point: ArrayLike, 
                            axtols: ArrayLike = np.array(0)) -> list[DisLinearPrecond]:
    """
    Given a list of constraints for the positive region and a point, return a list precondition pulling
    the positive region back over relu, making sure the given point is contained in the returned
    postcondition. The `axtols` gives bounds for each axis so that if `point > axtols` holds, the
    returned postcondition does not have any negative side behavior. May return an empty list if no
    appropriate preconditions are found.
    """
    # TODO remove
    assert point.ndim == 1
    for m, b in zip(ms, bs):
        assert m.ndim == 2
        assert b.ndim == 1
        assert m.shape[1] == b.shape[0]
        assert m.shape[0] == point.shape[0]
    
    out_l = []
    
    # Special case when point is in all positive quadrant + tolerance
    if np.all(point > axtols):
        for m, b in zip(ms, bs):
            if np.all(point @ m <= b):
                out_l.append(DisLinearPrecond(m, b, neg_side_type = NegSideType.NONE))
        return out_l
    
    # General case, prepare for quad pullback
    ap = np.all(point >= 0)
    quad = np.where(point <= 0)[0] if not ap else np.where(point <= axtols)[0]
    for m, b in zip(ms, bs):
        
        # If zero is in the region, try pulling back around it.
        if np.all(b > FLOAT_ATOL):
            c = DisLinearPrecond(m, b, neg_side_type = NegSideType.ZERO)
            if np.all(point[c.zer_i] <= c.zer_b) or (ap and np.all(point @ m <= b)):
                out_l.append(c)
                continue
        
        # Else, do quad pullback
        c = DisLinearPrecond(m, b, neg_side_type = NegSideType.QUAD, neg_side_quad = quad)
        if ( np.all(point @ m <= b) if ap else np.all(point @ c.neg_m <= c.neg_b) ):
            out_l.append(c)

    return out_l


def pull_back_precond_linear(prec: DisLinearPrecond, weights: ArrayLike, biases: ArrayLike 
                                                ) -> tuple[list[ArrayLike], list[ArrayLike]]:
    """
    Pulls back given DisLinearPrecond across a linear layer, and returns a lit of lps. Each lp is
    given by x @ m <= b, and the list of ms and bs are returned.
    """
    # TODO remove
    assert biases.ndim == 1
    assert weights.ndim == 2
    assert weights.shape[1] == biases.shape[0]
    assert weights.shape[1] == prec.num_neuron
    
    # Extract the positive region region
    m, b = prec.get_pos_constrs()
    ms = [m]
    bs = [b]
    
    # Extract the negative side region if it is there
    r = prec.get_neg_constrs()
    if r is not None:
        ms.append(r[0])
        bs.append(r[1])
    
    # Pull both back across linear transform
    for i in range(len(ms)):
        bs[i] -= biases @ ms[i]
        ms[i] = weights @ ms[i]
    
    # Pull back over relu and return
    return ms, bs
    

if __name__ == "__main__":  #DEBUG
    
    from timeit import timeit
    import random
    import sys
    
    from debug import rand_sparce_matrix, rand_sparce_pos_matrix
    

    n = 200         # Number of neurons in prev layer
    m = 160         # " "               in current layer
    k = 300         # Number of equations in postcond
    poivar = 10     # Range of values for the center point
    biavar = 10     # Range of bias values
    p0 = random.random() * 0.1
    p1 = random.random() * 0.15
    n_run = 1000
    
    t = 0

    mat, bnd, pt, weights, bias = None, None, None, None, None
    n_succ = 0
    n_none = 0
    n_zero = 0
    n_quad = 0
    
    def fail_dump():
        global n, k, basis, center, tc1, tc2
        # Dump basis, center and tie classes found to log if methods do not match.
        data = {}
        data['weights'] = weights.tolist()
        data['bias'] = bias.tolist()
        data['mat'] = mat.tolist()
        data['bnd'] = bnd.tolist()
        data['pt'] = pt.tolist()
        with open(sys.argv[1], 'w') as log:
            log.write(str(data))
        
    def run_pb():
        global mat, bnd, pt, weights, bias, n_succ, n_none, n_zero, n_quad
        log("Running pullback")
        ret = pull_back_precond(DisLinearPrecond(mat, bnd), weights, bias, pt)
        n_succ += len(ret)
        for r in ret:
            if r.neg_side_type == NegSideType.NONE:
                n_none += 1
            elif r.neg_side_type == NegSideType.ZERO:
                n_zer += 1
            elif r.neg_side_type == NegSideType.QUAD:
                n_quad += 1
    
    if len(sys.argv) >= 3 and sys.argv[2] == "checklog":
        with open(sys.argv[1]) as log:
            data = eval(log.read())
            weights = np.array(data['weights'])
            bias = np.array(data['bias'])
            mat = np.array(data['mat'])
            bnd = np.array(data['bnd'])
            pt = np.array(data['pt'])
            
            log("Running pullback")
            try:
                t += timeit(run_pb, number=1)
            except Exception as e:
                fail_dump()
                raise e
            
            exit()
            
    
    for i in range(n_run):
        log(f"Run {i} of {n_run}")

        log("Generating data")
        mat = rand_sparce_matrix(m,k,2*p0)
        pt = (np.random.rand(n) - 0.5) * poivar
        weights = rand_sparce_pos_matrix(n,m,p0)
        bias = (np.random.rand(m) - 0.5) * biavar
        p_ = np.copy(pt)
        p_[ np.where(p_ < 0) ] = 0
        bnd = (p_ @ weights + bias) @ mat + 1
        
        log("Running pullback")
        try:
            t += timeit(run_pb, number=1)
        except Exception as e:
            fail_dump()
            raise e

    t /= n_run
    log(f"The average time for pullback is {t}")
    log(f"The layer went from {n} to {m} neurons and the right precondition had {k} constraints")
    log(f"There were {n_succ} pullbacks, {n_none} NONE, {n_zero} ZERO, {n_quad} QUAD")
