"""
The entry point of the code, also handles multiprocess concurrency.
"""


from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from concurrency import init, start, stop, add_task, get_return, log
from postcondition import LinearPostcond, push_forward_postcond
from precondition import DisLinearPrecond, pull_back_constr_relu, pull_back_precond
from inclusion import check_inclusion



class TaskMessage(Enum):
    """
    Records the kind of task messages that can be sent to a worker's kernel. Each message asks the
    worker to do a particular task. Task messages are of the follwing form: (TaskMessage, layer, *args)
    """
    PUSH_FORD       = 1     # Push forward precondition
    PULL_BACK       = 2     # Pull back postcondition
    PULL_BACK_OUTLP = 3     # Pull the output lp back across final ReLu
    INCLUSION_CHK   = 4     # Check inclusion of a postcondition in a precondtion

    
    
class ReturnMessage(Enum):
    """
    The message passed when a task returns something. The message structure is (ReturnMessage,
    layer, data)
    """
    PUSH_F_DONE     = 0     # Pushforward has completed
    PULL_B_DONE     = 1     # Pullback done, used both for the output lp and in intermediate layers
    INCL_CHK_DONE   = 2     # Checking inclusion is done


class PermCheckReturnKind(Enum):
    """
    The kind of a return from the algorithm
    """
    PROOF           = 'proof'
    COUNTEREXAMPLE  = 'counterexample'
    INCONCLUSIVE    = 'inconclusive'


class PermCheckReturnStruct:
    """
    A class containing a proof or counterexample, returned when algo exits

    Members:
    
    kind            -   The kind of return that happened, proof, cex or inconclusive
    
    If the kind is 'proof', the follwing are present:
    
    inp_basis,
    inp_center      -   Basis and center for the input.
    postconds       -   A list of postconditions leading up to the layer where inclusion was found
    out_lp_m,
    out_lp_b        -   Convex polytope of the safe region in the output.
    preconds        -   A list of preconditions from the layer where the postcondition was found
    incl_layer      -   The layer at which the inclusion was found
    
    If the kind is 'counterexample', the following are present:
    
    counterexamples -   A list of counterexamples. Each counterexample should be a 5-tuple: the
                        input, the permuted input, the output produced, the output permuted
                        according to the output permutation and the ouput produced from the
                        permuted input.
    """
    def __init__(self, kind : PermCheckReturnKind, *args):
        """
        Args should init other members depending on what kind is, see above. 
        """
        self.kind = kind
        
        if self.kind == PermCheckReturnKind.PROOF:
            (self.inp_basis, self.inp_center, self.postconds, 
                self.out_lp_m, self.out_lp_b, self.preconds, self.incl_layer) = args
            
        elif self.kind == PermCheckReturnKind.COUNTEREXAMPLE:
            self.counterexamples = args[0]

    def __str__(self):
        """
        Print out a short summary of the proof or counterexample situation
        """
        if self.kind == PermCheckReturnKind.INCONCLUSIVE:
            return "PermCheck has returned INCONCLUSIVE"
        
        elif self.kind == PermCheckReturnKind.COUNTEREXAMPLE:
            return "PermCheck has found {0} COUNTEREXAMPLES".format(len(self.counterexamples))
        
        elif self.kind == PermCheckReturnKind.PROOF:
            return "PermCheck has successfully PROVED via inclusion at layer {0}".format(
                                                                            self.incl_layer)

    def __repr__(self):
        """
        Print all details of proof, or all counterexamples.
        """
        if self.kind == PermCheckReturnKind.INCONCLUSIVE:
            return "PermCheck has returned INCONCLUSIVE"

        elif self.kind == PermCheckReturnKind.COUNTEREXAMPLE:
            s = "PermCheck has found {0} COUNTEREXAMPLES: \n".format(len(self.counterexamples))
            for i, (x, sx, nx, snx, nsx) in enumerate(self.counterexamples()):
                s += "\nCounterexample {0}:\n".format(i)
                s += "    Input:                            {0}\n".format(x)
                s += "    Permuted Input:                   {0}\n".format(sx)
                s += "    Output from Input:                {0}\n".format(nx)
                s += "    Permutation of Output from Input: {0}\n".format(snx)
                s += "    Output from Permuted Input:       {0}\n".format(nsx)
            return s
        
        elif self.kind == PermCheckReturnKind.PROOF:
            s = "PermCheck has successfully PROVED via inclusion at layer {0}: \n".format(
                                                                            self.incl_layer)
            
            s += "\nThe input joint vectors are given by:\n a @ {0} + {1}\n".format( self.inp_basis,
                                                                                    self.inp_center)
            for i, postc in enumerate(self.postconds):
                s += "\nWhich leads to values after the linear layer {0} of form:\n".format(i)
                s += "a @ {0} + {1}\n".format(postc.basis, postc.center)
            
            for i, prec in enumerate(self.preconds):
                l = self.incl_layer + i
                pm, pb = prec.get_pos_constrs()
                s += "\nEach of are values after the linear layer {0} that satisfy:\n".format(l)
                s += "x @ {0} <= {1}\n".format(pm, pb)
                r = prec.get_neg_constrs()
                if r is not None:
                    nm, nb = r
                    s += "    Or satisfy:\n"
                    s += "x @ {0} <= {1}\n".format(nm, nb)
                    
            s += "\nWhich finally satisfy the LP:\n x @ {0} <= {1}\n".format(self.out_lp_m,
                                                                        self.out_lp_b)
            s += "\nWhich is characterizes the output condition"

            return s
                
        



def kernel( *args): #task_id : TaskMessage, layer : int, *args : Any ) -> Any:
    """
    The primary kernel that runs in the worker threads and dispatches the commands.
    """
    task_id, layer, *args = args
    log("Acquired task {0}".format(task_id))
    
    # Run pushforward
    if task_id == TaskMessage.PUSH_FORD:
        postc, weights, bias = args
        log("Pushing forward postcondition across layer {0}".format(layer)) 
        ret = push_forward_postcond(postc, weights, bias)
        log("Push forward done")
        return ReturnMessage.PUSH_F_DONE, layer, ret

    # Pull back the output LP across the final ReLu
    elif task_id == TaskMessage.PULL_BACK_OUTLP:
        ms, bs, centr = args
        log("Pulling back output LP across last layer, {0}, {1}".format(ms.shape, bs.shape)) 
        ret = pull_back_constr_relu([ms], [bs], centr)
        log("Pullback done")
        log("Returning {0} from outlp".format(ret)) #DEBUG
        return ReturnMessage.PULL_B_DONE, layer, ret
    
    # Pull back precond over layer
    elif task_id == TaskMessage.PULL_BACK:
        prec, weights, bias, centr = args
        log("Pulling back over layer {0}".format(layer))
        ret = pull_back_precond(prec, weights, bias, centr)
        log("Pullback done")
        log("Returning {0}".format(ret)) #DEBUG
        return ReturnMessage.PULL_B_DONE, layer, ret

    # Inclusion check
    elif task_id == TaskMessage.INCLUSION_CHK:
        postc, prec = args
        log("Cheking inclusion at {0}".format(layer))
        cexes = check_inclusion( postc, prec )
        log("Done checking inclusion")
        return ReturnMessage.INCL_CHK_DONE, layer, cexes
    
    else:
        log("Unknown Task {0}".format(task_id))




def main(   weights : list[ArrayLike], biases : list[ArrayLike],
            pre_perm : list[int], pre_lb : ArrayLike, pre_ub : ArrayLike,
            post_perm : list[int], post_epsilon : float) -> None:
    """
    The main method.
    
    DNN positions are indexed as follows:
   
   
                        pre[0], post[0]                 pre[1], post[1]   pre[..], post[..]
                                V                             V               V
    Input -> weight[0], bias[0] -> Relu -> weight[1], bias[1] -> Relu -> .... -> Relu -> Output
            |                          |  |                          |
            +--------- Layer 0 --------+  +------- Layer 1 ----------+
            
    
    Arguments:
    
    weights, biases     -   A list of the weights and biases of the DNN. If the weights take n
                            neurons to m neurons, the weight matrix should have shape (n, m)
    pre_perm, post_perm -   The pre and post permutations. Permutations are given as lists of
                            integers, so that the i-th integer is the position where i is mapped to
    pre_ub, pre_lb      -   The upper and lower bounds within which the input may lie, both are
                            ArrayLikes
    post_epsilon        -   The tolerance for being the output being within the permutation
    """
    
    # Initialize workers
    init(kernel)


    # Common vars
    n_inputs = weights[0].shape[0]
    n_outputs = biases[-1].shape[0]
    n_layers = len(weights)
    postconds = [ None ] * n_layers               # Pre and post conditions
    preconds  = [ None ] * n_layers
    centers = [ None ] * n_layers               # Center points
    out_lp_m = out_lp_b = None                  # The LP on the final output side
    cexes = [ [] for _ in range(n_layers) ]
 
 
    # Check if dimensionalities are correct
    for i in range(n_layers-1):
        if weights[i].ndim != 2:
            raise ValueError("Weights {1} must be a matrix, {0}".format(weights[i].shape, i))
        if biases[i].ndim != 1:
            raise ValueError("Bias must be a vector")
        if weights[i].shape[1] != biases[i].shape[0]:
            raise ValueError("Weight and biases shape mismatch at layer {0}".format(i))
        if weights[i+1].shape[0] != biases[i].shape[0]:
            raise ValueError("Weight and biases shape mismatch between layer {0} and {1}".format(i, i+1))
    if weights[-1].shape[1] != biases[-1].shape[0]:
        raise ValueError("Weight and biases shape mismatch at layer {0}".format(n_layers-1))
    
  
    # Duplicate weights and biases for permuted and unpermuted vars
    weights = [ np.block(  [[ w, np.zeros(w.shape) ],
                            [ np.zeros(w.shape), w ]] ) for w in weights ]
    biases = [  np.concatenate((b, b)) for b in biases ]
    
    log("Doubled up weights {0} and biases {1}".format(weights, biases))

    # Generate first postcondition
    brn = (pre_ub - pre_lb) / 2
    cen = (pre_ub + pre_lb) / 2
    brn = np.concatenate((brn, brn[pre_perm]))
    inp_c = np.concatenate((cen, cen[pre_perm]))
    inp_b = np.block([ np.eye(n_inputs), np.zeros((n_inputs, n_inputs)) ])
    for i, p in enumerate(pre_perm):        # Permutations
        inp_b[i, n_inputs + p] = -1
    inp_b *= brn[np.newaxis, :]             # Scale to fill bounds
    log("Inp_c: {0}".format(inp_c.shape))
    postconds[0] = LinearPostcond(inp_b @ weights[0], inp_c @ weights[0] + biases[0])
    
    log("Starting algo for {0} layers".format(n_layers))
    
    # Start workers, que up pushforward
    start()
    add_task( (TaskMessage.PUSH_FORD, 0, postconds[0], weights[1], biases[1]) )
    
    # Push forward center point
    centers[0] = inp_c @ weights[0] + biases[0]
    for i in range(1, n_layers):
        rc = np.copy(centers[i-1])
        rc[ np.where( rc < 0 )] = 0
        centers[i] = rc @ weights[i] + biases[i]
    
    
    # Set up LP for the output condition
    p_blk = np.zeros((n_outputs, n_outputs))
    for i, p in enumerate(post_perm):
        p_blk[p, i] = 1
    out_lp_m = np.block([[ np.eye(n_outputs), -np.eye(n_outputs) ], [ -p_blk, p_blk,  ]])
    out_lp_b = np.ones(2 * n_outputs) * post_epsilon
    
    
    # Schedule pullback of out lp
    add_task( (TaskMessage.PULL_BACK_OUTLP, n_layers, out_lp_m, out_lp_b, centers[-1]) )
    
    
    # Start loop acting on returned messages
    pf_remaining = True         # Is pushforwards all done?
    pb_remaining = True         # Is pullbacks all done
    n_incl_check = 0            # Number of inclusion checks performed
    while pf_remaining or pb_remaining or n_incl_check < n_layers:
        
        # Get returned message
        msg, layer, *data = get_return(wait=True)
        log("Recieved message {0} at layer {1}".format(msg, layer))
       
       
        # If message says a pushforward is done, que the next if available, or set flags
        if msg == ReturnMessage.PUSH_F_DONE:

            # Get the postcond
            postconds[layer+1] = data[0]
            
            log("Added postcond {0}".format(data[0]))   # DEBUG
            log("Layer {0} has postcond of dim {1}".format(layer+1, data[0].num_neuron))
            
            # If we also have a precondtion, schedule an inclusion check
            if preconds[layer+1] is not None:
                add_task( (TaskMessage.INCLUSION_CHK, layer+1, postconds[layer+1], preconds[layer+1]) )

            # Early loop around of no pushforwards remaining
            if layer+2 >= n_layers:       
                log("All pushforwards complete")
                pf_remaining = False
                continue
            
            # Schedule next pullback
            log("Scheuling pushforward across layer {0}".format(layer+1))
            add_task( (TaskMessage.PUSH_FORD, layer+1, postconds[layer+1], weights[layer+2], 
                        biases[layer+2]) )
        
        
        # If the pullback is done, schedule next one, set flags, and schedule an inclusion check.
        elif msg == ReturnMessage.PULL_B_DONE:
            
            # Get the precond, quit if none found
            if len(data[0]) > 0:
                preconds[layer-1] = data[0][0]      # For now, just pick the first precond produced
            else:
                stop()
                return PermCheckReturnStruct( PermCheckReturnKind.INCONCLUSIVE )
            
            log("Added precond {0}".format(data[0][0]))   # DEBUG
            log("Layer {0} has precond of dim {1}".format(layer-1, data[0][0].num_neuron))

            # If we also have a precondtion, schedule an inclusion check
            if postconds[layer-1] is not None:
                add_task( (TaskMessage.INCLUSION_CHK, layer-1, postconds[layer-1], preconds[layer-1]) )

            # Early loop around of no pushforwards remaining
            if layer-1 <= 0:       
                log("All pullbacks complete")
                pb_remaining = False
                continue
            
            # Schedule next pullback
            log("Scheuling pullback across layer {0}".format(layer-1))
            add_task( (TaskMessage.PULL_BACK, layer-1, preconds[layer-1], weights[layer-1], 
                        biases[layer-1], centers[layer-2]) )
                
        # If the inclusion check is done, quit if successfull, or try to pull back cex
        elif msg == ReturnMessage.INCL_CHK_DONE:
            
            cexes = data[0]
            
            # Quit out and return if there are no cexes.
            if len(cexes) == 0:  
                log("Found proof via inclusion at layer {0}".format(layer))
                stop()
                return PermCheckReturnStruct( PermCheckReturnKind.PROOF, 
                        inp_b, inp_c, postconds[:layer+1], out_lp_m, out_lp_b, preconds[layer:],
                        layer)
            
            n_incl_check += 1
            
            log("No inclusion at layer {0}, found {2} cexes, checked {1} layers".format(
                        layer, n_incl_check, len(cexes)))
            

        else:
            log("Unknown return message {0}".format(msg))
  
    
    log("Main process coordination loop has stopped, flags are {0}, {1}, {2}".format(
                pf_remaining, pb_remaining, n_incl_check))
   
    # If we failed to find a proof or a cex, return inconclusive
    stop()
    return PermCheckReturnStruct( PermCheckReturnKind.INCONCLUSIVE )
   
   
   
    
if __name__ == "__main__":

    weights = [ np.array(   [[1000, -1000, 1000, -1000],
                             [-1000, 1000, -1000, 1000]] ),
                np.array(   [[1, 0], [0, 1], [-1, 0], [0, -1]] )]
    biases = [ np.array( [0, 0, -1, -1] ), np.array( [0, 0] ) ]
    sig = [1, 0]
    ret = main(weights, biases, sig, np.array([0, 0]), np.array([1,1]), sig, 0.1)
    print("Returned {1}: {0}".format(str(ret), ret.kind.value))
    
    print("Details:")
    print(repr(ret))
