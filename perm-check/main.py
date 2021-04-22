"""
The entry point of the code, also handles multiprocess concurrency.
"""


from enum import Enum, unique
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from concurrency import init, start, stop, add_task, get_return, log
from postcondition import LinearPostcond, push_forward_postcond_linear, push_forward_postcond_relu
from precondition import DisLinearPrecond, pull_back_constr_relu, pull_back_precond_linear
from inclusion import check_inclusion_pre_linear, check_inclusion_pre_relu
from global_consts import MP_NUM_WORKER


@unique
class TaskMessage(Enum):
    """
    Records the kind of task messages that can be sent to a worker's kernel. Each message asks the
    worker to do a particular task. Task messages are of the follwing form: (TaskMessage, layer, *args)
    """
    PUSH_F_LINEAR   = 1     # Push forward precondition across a linear layer. Expects postcond,
                            # weights, and bias.
    PUSH_F_RELU     = 2     # Push forward precondition across a ReLU layer. Expects postcond.
    PULL_B_LINEAR   = 3     # Pull back postcondition across linear layer. Expects a
                            # DisLinearPrecond, weights and bias
    PULL_B_RELU     = 4     # Pull the output lp back across ReLU. Expects an LP as a (m,b) pair,
                            # and a center point
    ICHK_PRE_LINEAR = 5     # Check inclusion of postcond into the precond at a position before a
                            # linear layer. Expects a LinearPostcond and an lp given by a (m,b)
                            # pair.
    ICHK_PRE_RELU   = 6     # Check inclusion of a postcond in a precond at position just before
                            # ReLU. Expects a LinearPostcond and a DisLinearPrecond

    
@unique
class ReturnMessage(Enum):
    """
    The message passed when a task returns something. The message structure is (ReturnMessage,
    layer, data)
    """
    PUSH_F_LINEAR_DONE  = 0     # Pushforward across linear layer has completed. Contains postcond
                                # on other side
    PUSH_F_RELU_DONE    = 1     # Pushforward across ReLU layer has completed. Contains postcond on
                                # other side
    PULL_B_LINEAR_DONE  = 2     # Pullback across linear layer done. Contains an lp as (m,b) pair.
    PULL_B_RELU_DONE    = 3     # Pullback across ReLU done. Contains a DisLinearPrecond, or none
    ICHK_PRELIN_DONE    = 4     # Checking inclusion at position before a linear layer is done.
                                # Contains a list of counterexamples.
    ICHK_PREREL_DONE    = 5     # Checking inclusion at position before a relu layer is done.
                                # Contains a list of counterexamples.


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
    
    kind                -   The kind of return that happened, proof, cex or inconclusive
    
    If the kind is 'proof', the follwing are present:
    
    postconds           -   A list of postconditions leading up to the layer where inclusion was
                            found
    pre_relu_pconds,    -   A list of preconditions from the layer where the inclusion was found
    pre_linear_pconds
    incl_layer          -   The layer at which the inclusion was found
    
    If the kind is 'counterexample', the following are present:
    
    counterexamples     -   A list of counterexamples. Each counterexample should be a 5-tuple: the
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
            self.postconds, self.pre_relu_pconds, self.pre_linear_pconds, self.incl_layer = args
            
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
            
            # Print the first postcondition
            s += "\nThe input joint vectors are given by:\n a @ {0} + {1}\n".format(
                                            self.postconds[0].basis, self.postconds[0].center )
            

            # Print all the postconditions upto the inclusion layer
            l = 0
            while 2*l + 2 < len(self.postconds):
                
                # After linear layer
                s += "\nWhich leads to values after the linear layer {0} of form:\n".format(l)
                s += "a @ {0} + {1}\n".format(self.postconds[2*l+1].basis, self.postconds[2*l+1].center)

                # After ReLU
                s += "\nWhich leads to values after the ReLU layer {0} of form:\n".format(l)
                s += "a @ {0} + {1}\n".format(self.postconds[2*l+2].basis, self.postconds[2*l+2].center)
                l += 1
                
                
            # Handle inclusion before between relu and linear layers
            prlu_pcs = self.pre_relu_pconds
            l = self.incl_layer
            if 2*l + 1 < len(self.postconds):
                
                # Print postcondition
                s += "\nWhich leads to values after the linear layer {0} of form:\n".format(l)
                s += "a @ {0} + {1}\n".format(self.postconds[2*l+1].basis, self.postconds[2*l+1].center)
                
                # Print precondition
                pm, pb = prlu_pcs[0].get_pos_constrs()
                s += "\nEach of which satisfy:\n"
                s += "x @ {0} <= {1}\n".format(pm, pb)
                r = prec.get_neg_constrs()
                if r is not None:
                    nm, nb = r
                    s += "    Or satisfy:\n"
                    s += "x @ {0} <= {1}\n".format(nm, nb)
                
                # Adjust pre-relu preconds
                prlu_pcs = prlu_pcs[1:]
                l += 1
                

            # Print the preconditions
            for prlu, plin in zip(prlu_pcs, self.pre_linear_pconds):
                
                # Print the pre-linear precond
                m, b = plin
                s += "\nEach of which give values before the linear layer {0} that satisfy:\n".format(l)
                s += "x @ {0} <= {1}\n".format(m, b)
                
                # Print the pre-relu precond
                pm, pb = prec.get_pos_constrs()
                s += "\nEach of are values after the linear layer {0} that satisfy:\n".format(l)
                s += "x @ {0} <= {1}\n".format(pm, pb)
                r = prec.get_neg_constrs()
                if r is not None:
                    nm, nb = r
                    s += "    Or satisfy:\n"
                    s += "x @ {0} <= {1}\n".format(nm, nb)

                l += 1
                    

            # Print the output condition
            m, b = self.pre_linear_pconds[-1]
            s += "\nEach of which give values before the linear layer {0} that satisfy:\n".format(l)
            s += "x @ {0} <= {1}\n".format(m, b)
            
            s += "\nWhich characterizes the output condition"

            return s
                
        



def kernel( *args): #task_id : TaskMessage, layer : int, *args : Any ) -> Any:
    """
    The primary kernel that runs in the worker threads and dispatches the commands.
    """
    task_id, layer, *args = args
    log("Acquired task {0}".format(task_id))
    
    # Run pushforward across linear
    if task_id == TaskMessage.PUSH_F_LINEAR:
        postc, weights, bias = args
        log("Pushing forward postcondition across linear layer {0}".format(layer)) 
        ret = push_forward_postcond_linear(postc, weights, bias)
        log("Push forward across linear done")
        return ReturnMessage.PUSH_F_LINEAR_DONE, layer, ret
    
    # Run pushforward across Relu
    if task_id == TaskMessage.PUSH_F_RELU:
        postc = args[0]
        log("Pushing forward postcondition across relu layer {0}".format(layer)) 
        ret = push_forward_postcond_relu(postc)
        log("Push forward across relu done")
        return ReturnMessage.PUSH_F_RELU_DONE, layer, ret

    # Pull back LP across ReLU
    elif task_id == TaskMessage.PULL_B_RELU:
        ms, bs, centr = args
        log("Pulling back output LP across relu layer {0}".format(layer)) 
        ret = pull_back_constr_relu([ms], [bs], centr)
        log("Pullback done")
        log("Returning {0} from outlp".format(ret)) #DEBUG
        return ReturnMessage.PULL_B_RELU_DONE, layer, ( ret[0] if len(ret) > 0 else None )
                                                # Return first precond, TODO refine
    
    # Pull back precond over linear layer
    elif task_id == TaskMessage.PULL_B_LINEAR:
        prec, weights, bias = args
        log("Pulling back over linear layer {0}".format(layer))
        ms, bs = pull_back_precond_linear(prec, weights, bias)
        log("Pullback done")
        return ReturnMessage.PULL_B_LINEAR_DONE, layer, (ms[0], bs[0]) # Choose positive, TODO refine

    # Inclusion check just before ReLU
    elif task_id == TaskMessage.ICHK_PRE_RELU:
        postc, prec = args
        log("Cheking inclusion at {0} before the ReLU".format(layer))
        log("Postcond {0}, precond {1}".format(postc, prec))
        cexes = check_inclusion_pre_relu( postc, prec )
        log("Done checking inclusion, found {0} cexex".format(len(cexes)))
        return ReturnMessage.ICHK_PREREL_DONE, layer, cexes

    # Inclusion check just before linear
    elif task_id == TaskMessage.ICHK_PRE_LINEAR:
        postc, m, b = args
        log("Cheking inclusion at {0} before the ReLU".format(layer))
        log("Postcond {0}, m {1}, b{2}".format(postc, m, b))
        cexes = check_inclusion_pre_linear( postc, m, b )
        log("Done checking inclusion, found {0} cexex".format(len(cexes)))
        return ReturnMessage.ICHK_PRELIN_DONE, layer, cexes
    
    else:
        log("Unknown Task {0}".format(task_id))




def main(   weights : list[ArrayLike], biases : list[ArrayLike],
            pre_perm : list[int], pre_lb : ArrayLike, pre_ub : ArrayLike,
            post_perm : list[int], post_epsilon : float,
            num_workers : int = MP_NUM_WORKER ) -> None:
    """
    The main method.
    
    DNN positions are indexed as follows:
   
   
        pos 0                 pos 1   pos 2                 pos 3   pos 4   pos ..   pos ..
          V                     V       V                     V       V       V       V
    Input -> weight[0], bias[0] -> Relu -> weight[1], bias[1] -> Relu -> .... -> Relu -> Output
          ^ |                   ^      |^ |                   ^      |^                
          | +--------- Layer 0 -|------+| +------- Layer 1 ---|------+|                
          |                     |       |                     |       |                
          |                     |       |                     |       |                
      pre_linear 0              | pre_linear 1                |  pre_linear 2  ....
                                |                             |
                            pre_relu 0                      pre_relu 1 ....
    
    Arguments:
    
    weights, biases     -   A list of the weights and biases of the DNN. If the weights take n
                            neurons to m neurons, the weight matrix should have shape (n, m)
    pre_perm, post_perm -   The pre and post permutations. Permutations are given as lists of
                            integers, so that the i-th integer is the position where i is mapped to
    pre_ub, pre_lb      -   The upper and lower bounds within which the input may lie, both are
                            ArrayLikes
    post_epsilon        -   The tolerance for being the output being within the permutation
    num_workers         -   The number of worker processes to pass
    """
    
    # Initialize workers
    init(kernel, n_workers = num_workers)


    # Common vars
    n_inputs = weights[0].shape[0]
    n_outputs = biases[-1].shape[0]
    n_layers = len(weights)
    n_pos = 2*n_layers + 1
    
    postconds   = [ None ] * n_pos              # Pre and post conditions
    centers     = [ None ] * n_pos              # Center points
    
    pre_relu_pconds     = [ None ] * n_layers       # The preconditions for the positions just
    pre_linear_pconds   = [ None ] * (n_layers + 1) # before and after ReLU have seperate types and
                                                    # are stored seperately. `pre_relu` are
                                                    # DisLinearPreconds, `pre_linear` are LPs given
                                                    # by (m, b) pairs
    
    cexes = [ [] for _ in range(n_pos) ]
 
 
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
        inp_b[i, n_inputs + p] = 1
    inp_b *= brn[np.newaxis, :]             # Scale to fill bounds
    log("Inp_c: {0}".format(inp_c.shape))
    postconds[0] = LinearPostcond(inp_b, inp_c)
    
    log("Starting algo for {0} layers".format(n_layers))
    
    # Start workers, que up pushforward
    start()
    add_task( (TaskMessage.PUSH_F_LINEAR, 0, postconds[0], weights[0], biases[0]) )
    
    # Push forward center point
    centers[0] = inp_c
    for i in range(n_layers):
        centers[ 2*i + 1 ] = centers[2 * i] @ weights[i] + biases[i]    # Through linear
        centers[ 2*i + 2 ] = np.copy( centers[ 2*i + 1 ] )              
        centers[ 2*i + 2 ][ np.where( centers[ 2*i + 2 ] < 0 )] = 0     # Trhough relu
    
    
    # Set up output side postcondition
    p_blk = np.zeros((n_outputs, n_outputs))
    for i, p in enumerate(post_perm):
        p_blk[p, i] = 1
    out_lp_m = np.block([[ np.eye(n_outputs), -np.eye(n_outputs) ], [ -p_blk, p_blk,  ]])
    out_lp_b = np.ones(2 * n_outputs) * post_epsilon
    pre_linear_pconds[-1] = (out_lp_m, out_lp_b)
    
    
    # Schedule pullback of out lp
    add_task( (TaskMessage.PULL_B_RELU, n_layers, out_lp_m, out_lp_b, centers[-1]) )
    
    
    # Start loop acting on returned messages
    pf_remaining = True         # Is pushforwards all done?
    pb_remaining = True         # Is pullbacks all done
    n_incl_check = 0            # Number of inclusion checks performed
    while pf_remaining or pb_remaining or n_incl_check < n_pos:
        
        # Get returned message
        msg, layer, *data = get_return(wait=True)
        log("Recieved message {0} at layer {1}".format(msg, layer))
       
       
        # If message says a pushforward is done, que the next if available, or set flags
        if msg == ReturnMessage.PUSH_F_LINEAR_DONE:

            # Get the postcond
            postconds[2*layer+1] = data[0]
            
            log("Added pre-relu postcond {0}".format(data[0]))   # DEBUG
            log("Layer {0} has pre-relu postcond of dim {1}".format(layer, data[0].num_neuron))
            
            # If we also have a precondtion, schedule an inclusion check
            if pre_relu_pconds[layer] is not None:
                add_task( (TaskMessage.ICHK_PRE_RELU, layer, postconds[2*layer+1], 
                            pre_relu_pconds[layer]) )

            # Schedule next pullback
            log("Scheuling pushforward across relu layer {0}".format(layer))
            add_task( (TaskMessage.PUSH_F_RELU, layer, postconds[2*layer+1]) )
        
       
        # If message says a pushforward is done, que the next if available, or set flags
        if msg == ReturnMessage.PUSH_F_RELU_DONE:

            # Get the postcond
            pcidx = 2*(layer + 1)           # where the next postcond will go
            postconds[pcidx] = data[0]
            
            log("Added pre-linear postcond {0}".format(data[0]))   # DEBUG
            log("Layer {0} has pre-linear postcond of dim {1}".format(layer, data[0].num_neuron))
            
            # If we also have a precondtion, schedule an inclusion check
            if pre_linear_pconds[layer+1] is not None:
                m, b = pre_linear_pconds[layer+1]
                add_task( (TaskMessage.ICHK_PRE_LINEAR, layer+1, postconds[pcidx], m, b) )

            # Terminate if no further pushforward possible
            if pcidx+1 >= n_pos:       
                log("All pushforwards complete")
                pf_remaining = False
                continue
            
            # Schedule next pullback
            log("Scheuling pushforward across linear layer {0}".format(layer+1))
            add_task( (TaskMessage.PUSH_F_LINEAR, layer+1, postconds[pcidx], weights[layer+1], 
                        biases[layer+1]) )
       
            
        # If the pullback is done, schedule next one, set flags, and schedule an inclusion check.
        elif msg == ReturnMessage.PULL_B_RELU_DONE:
            
            # Quit if no pullback was found
            if data[0] is None:
                log("No pullback found at layer {0}".format(layer))
                stop()
                return PermCheckReturnStruct( PermCheckReturnKind.INCONCLUSIVE )
                
            # Get the precond
            pre_relu_pconds[layer-1] = data[0]      # For now, just pick the first precond produced
            
            log("Added pre-relu precond {0}".format(data[0]))   # DEBUG
            log("Layer {0} has pre-relu precond of dim {1}".format(layer-1, data[0].num_neuron))

            # If we also have a precondtion, schedule an inclusion check
            if postconds[2*layer-1] is not None:
                add_task( (TaskMessage.ICHK_PRE_RELU, layer-1, postconds[2*layer-1], 
                            pre_relu_pconds[layer-1]) )

            # Schedule next pullback
            log("Scheuling pullback across linear layer {0}".format(layer-1))
            add_task( (TaskMessage.PULL_B_LINEAR, layer-1, data[0], weights[layer-1], 
                        biases[layer-1]) )
           
            
        # If the pullback is done, schedule next one, set flags, and schedule an inclusion check.
        elif msg == ReturnMessage.PULL_B_LINEAR_DONE:
            
            pre_linear_pconds[layer] = data[0]      # For now, just pick the first precond produced
            
            log("Added pre-linear precond {0}".format(data[0]))   # DEBUG
            log("Layer {0} has pre-linear precond of shape {1}, center is {2}".format(layer-1,
                                                    data[0][0].shape, centers[layer*2].shape))

            # If we also have a precondtion, schedule an inclusion check
            if postconds[2*layer] is not None:
                add_task( (TaskMessage.ICHK_PRE_LINEAR, layer, postconds[2*layer], data[0][0], 
                                                                                    data[0][1]) )

            # Terminate if no pullbacks remaining
            if layer <= 0:       
                log("All pullbacks complete")
                pb_remaining = False
                continue
            
            # Schedule next pullback
            log("Scheuling pullback across relu layer {0}".format(layer-1))
            add_task( (TaskMessage.PULL_B_RELU, layer, data[0][0], data[0][1], centers[layer*2]) )
           
            
        # If the inclusion check is done, quit if successfull, or try to pull back cex TODO cex pb
        elif msg == ReturnMessage.ICHK_PRELIN_DONE:
            
            cexes = data[0]
            
            # Quit out and return if there are no cexes.
            if len(cexes) == 0:  
                log("Found proof via inclusion just before layer {0}".format(layer))
                stop()
                return PermCheckReturnStruct( PermCheckReturnKind.PROOF, 
                        postconds[:2*layer+1], pre_relu_pconds[layer:], pre_linear_pconds[layer:],
                        layer)
            
            n_incl_check += 1
            
            log("No inclusion at layer {0}, found {2} cexes, checked {1} layers".format(
                        layer, n_incl_check, len(cexes)))
            
            
        # If the inclusion check is done, quit if successfull, or try to pull back cex TODO cex pb
        elif msg == ReturnMessage.ICHK_PREREL_DONE:
            
            cexes = data[0]
            
            # Quit out and return if there are no cexes.
            if len(cexes) == 0:  
                log("Found proof via inclusion just before ReLU layer {0}".format(layer))
                stop()
                return PermCheckReturnStruct( PermCheckReturnKind.PROOF, 
                        postconds[:(layer+1)*2], pre_relu_pconds[layer:],
                        pre_linear_pconds[layer+1:], layer)
            
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

    import sys
    
    if len(sys.argv) < 2:
        print("Usage: `python main.py <number_of_worker_processes>`")
        exit(-1)
    
    weights = [ np.array(   [[1000, -1000, 1000, -1000],
                             [-1000, 1000, -1000, 1000]] ),
                np.array(   [[1, 0], [0, 1], [-1, 0], [0, -1]] )]
    biases = [ np.array( [0, 0, -1, -1] ), np.array( [0, 0] ) ]
    sig = [1, 0]
    
    ret = main(weights, biases, sig, np.array([0, 0]), np.array([1,1]), sig, 0.1,
                num_workers = int(sys.argv[1]))
    
    print("\n\nReturned {1}: {0}".format(str(ret), ret.kind.value))
    print("\n\nDetails:\n\n")
    print(repr(ret))
