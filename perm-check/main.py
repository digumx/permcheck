"""
The entry point of the code, also handles multiprocess concurrency.
"""


from enum import Enum, unique
from typing import Any, Union
from math import floor
from heapq import heappush, heappop
from traceback import format_exc
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from concurrency import init, start, stop, add_task, get_return, log, any_error
from postcondition import LinearPostcond, push_forward_postcond_linear, push_forward_postcond_relu
from precondition import DisLinearPrecond, pull_back_constr_relu, pull_back_precond_linear
from inclusion import check_inclusion_pre_linear, check_inclusion_pre_relu
from counterexample import pullback_cex, pullback_cex_linear, pullback_cex_relu
from global_consts import MP_NUM_WORKER, CEX_PULLBACK_NUM_TOTAL_CANDIDATES_MULT


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
    CEX_PB_LAYER    = 7     # Pull back a counterexample over a combined layer. Expects a
                            # counterexample, a LinearPostcond, weights, bias, and the number of
                            # counterexamples to return.
    CEX_PB_RELU     = 8     # Pull back a counterexample over a ReLU. Expects a counterexample and a
                            # LinearPostcond.
    CEX_PB_LINEAR   = 9     # Pull back a counterexample over a linear layer. Expects a
                            # counterexample, a LinearPostcond, weights and bias.

    
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
    CEX_PB_LAYER_DONE   = 6     # Done pulling back a cex over a layer. Contains a list of cexes
    CEX_PB_RELU_DONE    = 7     # Done pulling back a cex over a ReLU. Contains a single cex if
                                # found, else contains a None.
    CEX_PB_LINEAR_DONE  = 8     # Done pulling back a cex over a linear layer. Contains a single
                                # cex if found, else contains None.
                                

class PermCheckReturnKind(Enum):
    """
    The kind of a return from the algorithm
    """
    PROOF           = 'proof'
    COUNTEREXAMPLE  = 'counterexample'
    INCONCLUSIVE    = 'inconclusive'
    ERROR           = 'error'


class PermCheckReturnStruct:
    """
    A class containing a proof or counterexample, returned when algo exits

    Members:
    
    kind                -   The kind of return that happened, proof, cex or inconclusive
    time                -   The seconds taken to reach the conclusion
    
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

    If the kind is 'error', the following are present:
    
    trace               -   A traceback that caused the error, useful for debugging purposes
    """
    def __init__(self, kind : PermCheckReturnKind, time : float, *args):
        """
        Args should init other members depending on what kind is, see above. 
        """
        log("Constructing return struct")
        self.kind = kind
        self.time = time
        
        if self.kind == PermCheckReturnKind.PROOF:
            self.postconds, self.pre_relu_pconds, self.pre_linear_pconds, self.incl_layer = args
            
        elif self.kind == PermCheckReturnKind.COUNTEREXAMPLE:
            self.counterexamples = args[0]
            
        elif self.kind == PermCheckReturnKind.ERROR:
            self.trace = args[0]
            
        log("Return struct constructed")

    def __str__(self):
        """
        Print out a short summary of the proof or counterexample situation
        """
        if self.kind == PermCheckReturnKind.INCONCLUSIVE:
            return "PermCheck has returned INCONCLUSIVE in {0}".format( self.time )
        
        elif self.kind == PermCheckReturnKind.COUNTEREXAMPLE:
            return "PermCheck has found {0} COUNTEREXAMPLES in {1}".format(
                    len(self.counterexamples), self.time )
        
        elif self.kind == PermCheckReturnKind.PROOF:
            return "PermCheck has successfully PROVED via inclusion at layer {0} in {1}".format(
                                                                    self.incl_layer, self.time )
        elif self.kind == PermCheckReturnKind.ERROR:
            return "PermCheck has encountered an unhandled exception after {0}".format( self.time )

    def __repr__(self):
        """
        Print all details of proof, or all counterexamples.
        """
        if self.kind == PermCheckReturnKind.INCONCLUSIVE:
            return "PermCheck has returned INCONCLUSIVE in {0}".format( self.time )

        elif self.kind == PermCheckReturnKind.COUNTEREXAMPLE:
            s = "PermCheck has found {0} COUNTEREXAMPLES in {1}: \n".format(
                    len(self.counterexamples), self.time )
            for i, (x, sx, nx, snx, nsx) in enumerate(self.counterexamples):
                s += "\nCounterexample {0}:\n".format(i)
                s += "    Input:                            {0}\n".format(x.tolist())
                s += "    Permuted Input:                   {0}\n".format(sx.tolist())
                s += "    Output from Input:                {0}\n".format(nx.tolist())
                s += "    Permutation of Output from Input: {0}\n".format(snx.tolist())
                s += "    Output from Permuted Input:       {0}\n".format(nsx.tolist())
            return s
        
        elif self.kind == PermCheckReturnKind.PROOF:
            s = "PermCheck has successfully PROVED via inclusion at layer {0} in {1}: \n".format(
                                                                    self.incl_layer, self.time )
            
            # Print the first postcondition
            s += "\nThe input joint vectors are given by:\n a @ {0} + {1}\n".format(
                                            self.postconds[0].basis.tolist(),
                                            self.postconds[0].center.tolist() )
            

            # Print all the postconditions upto the inclusion layer
            l = 0
            while 2*l + 2 < len(self.postconds):
                
                # After linear layer
                s += "\nWhich leads to values after the linear layer {0} of form:\n".format(l)
                s += "a @ {0} + {1}\n".format(self.postconds[2*l+1].basis.tolist(),
                        self.postconds[2*l+1].center.tolist())

                # After ReLU
                s += "\nWhich leads to values after the ReLU layer {0} of form:\n".format(l)
                s += "a @ {0} + {1}\n".format(self.postconds[2*l+2].basis.tolist(),
                        self.postconds[2*l+2].center.tolist())
                l += 1
                
                
            # Handle inclusion before between relu and linear layers
            prlu_pcs = self.pre_relu_pconds
            l = self.incl_layer
            if 2*l + 1 < len(self.postconds):
                
                # Print postcondition
                s += "\nWhich leads to values after the linear layer {0} of form:\n".format(l)
                s += "a @ {0} + {1}\n".format(self.postconds[2*l+1].basis.tolist(),
                        self.postconds[2*l+1].center.tolist())
                
                # Print precondition
                pm, pb = prlu_pcs[0].get_pos_constrs()
                s += "\nEach of which satisfy:\n"
                s += "x @ {0} <= {1}\n".format(pm.tolist(), pb.tolist())
                r = prlu_pcs[0].get_neg_constrs()
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
                s += "x @ {0} <= {1}\n".format(m.tolist(), b.tolist())
                
                # Print the pre-relu precond
                pm, pb = prlu.get_pos_constrs()
                s += "\nEach of are values after the linear layer {0} that satisfy:\n".format(l)
                s += "x @ {0} <= {1}\n".format(pm.tolist(), pb.tolist())
                r = prlu.get_neg_constrs()
                if r is not None:
                    nm, nb = r
                    s += "    Or satisfy:\n"
                    s += "x @ {0} <= {1}\n".format(nm.tolist(), nb.tolist())

                l += 1
                    

            # Print the output condition
            m, b = self.pre_linear_pconds[-1]
            s += "\nEach of which give values before the linear layer {0} that satisfy:\n".format(l)
            s += "x @ {0} <= {1}\n".format(m.tolist(), b.tolist())
            
            s += "\nWhich characterizes the output condition"

            return s
        
        elif self.kind == PermCheckReturnKind.ERROR:
            return "\n\nPermCheck has encountered an unhandled exception in {1}: \n\n {0}".format(
                    self.trace, self.time )



class PriorityPreScheduler:
    """
    Uses a priority queue to schedule tasks according to a given priority, so that new tasks with a
    higher priority get performed before old tasks with a lower priority.

    Internally, it maintians a priority queue and a count of active tasks. When new tasks are added,
    they are added to the priority queue, and if the number of active tasks is less than a given
    maximum, the task with the highest priority is scheduled. Else, the task waits on the priority
    queue.     

    Members:
    _pqueue         -   Priority queue
    _n_scheduled    -   The number of currently scheduled tasks
    _nxt_tsk_idx    -   Index for the next task
    max_scheduled   -   Maximum allowed number of scheduled tasks
    """
    
    def __init__(self, max_scheduled):
        """
        Initialize with given maximum number of scheduled processes `max_scheduled`.
        """
        self.max_scheduled = max_scheduled
        self._n_scheduled = 0
        self._nxt_tsk_idx = 0
        self._pqueue = []
        
    def _schedule_next(self):
        """
        Attempt to schedule the next task from _pqueue.
        """
        # Early termination if too many already scheduled
        if self._n_scheduled >= self.max_scheduled:
            return
        
        # Early return if heap is empty
        if len(self._pqueue) <= 0:
            return

        # Schedule task
        self._n_scheduled += 1
        _, _, tsk = heappop(self._pqueue)
        add_task( tsk )
        log("Prescheduler: Scheduled task {0} across layer {1}, {2} already scheduled, {3} in que".format(
            tsk[0], tsk[1], self._n_scheduled, len(self._pqueue)))
        
    def task_done(self):
        """
        Called to notify the prescheduler that a scheduled task is complete
        """
        # Reduce the count
        self._n_scheduled -= 1
        if self._n_scheduled < 0: self._n_scheduled = 0
        log("Prescheduler: Task stopped, {0} scheduled, {1} waiting".format(self._n_scheduled, len(self._pqueue)))

        # Schedule next task
        self._schedule_next()
        
    def add_task(self, tsk, prio):
        """
        Called to preschedule the given task `tsk` with priority `prio`. The task is added to the
        priority queue
        """

        log("Prescheduler: Queing task {0} across layer {1} with priority {4}, {2} already scheduled, {3} in que".format(
            tsk[0], tsk[1], self._n_scheduled, len(self._pqueue), prio))
        
        # Set up task and add it to queue
        heappush(self._pqueue, (-prio, self._nxt_tsk_idx, tsk))
        self._nxt_tsk_idx += 1
        
        # Schedule next task
        self._schedule_next()

    def is_active(self):
        """
        Returns true if there are tasks waiting to be scheduled or scheduled tasks have not been
        completed
        """
        return self._n_scheduled > 0 or len(self._pqueue) > 0
        



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
        cexes = check_inclusion_pre_relu( postc, prec )
        log("Done checking inclusion, found {0} cexex".format(len(cexes)))
        return ReturnMessage.ICHK_PREREL_DONE, layer, cexes

    # Inclusion check just before linear
    elif task_id == TaskMessage.ICHK_PRE_LINEAR:
        postc, m, b = args
        log("Cheking inclusion at {0} before the ReLU".format(layer))
        cexes = check_inclusion_pre_linear( postc, m, b )
        log("Done checking inclusion, found {0} cexex".format(len(cexes)))
        return ReturnMessage.ICHK_PRELIN_DONE, layer, cexes

    # Pull back over a combined layer
    elif task_id == TaskMessage.CEX_PB_LAYER:
        cex, postc, w, b, npb = args
        log("Pulling back cex over combined layer {0}".format(layer))
        cexes = pullback_cex( cex, postc, w, b, npb )
        log("Done pulling back counterexample over combined layer {0}".format(layer))
        return ReturnMessage.CEX_PB_LAYER_DONE, layer, cexes

    # Pull back over a linear layer
    elif task_id == TaskMessage.CEX_PB_LINEAR:
        cex, postc, w, b = args
        log("Pulling back cex over linear layer {0}".format(layer))
        cex = pullback_cex_linear( cex, postc, w, b )
        log("Done pulling back counterexample over linear layer {0}".format(layer))
        return ReturnMessage.CEX_PB_LINEAR_DONE, layer, cex

    # Pull back over a relu layer
    elif task_id == TaskMessage.CEX_PB_RELU:
        cex, postc = args
        log("Pulling back cex over relu layer {0}".format(layer))
        cex = pullback_cex_relu( cex, postc )
        log("Done pulling back counterexample over relu layer {0}".format(layer))
        return ReturnMessage.CEX_PB_RELU_DONE, layer, cex
    
    else:
        log("Unknown Task {0}".format(task_id))



# DEBUG
n_cex_check_calls = 0

def check_cex( cex, weights, bias, out_lp_m, out_lp_b ):
    """
    Given a cex and a DNN, check if output satisfies x @ out_lp_m <= out_lp_b. If this is a true
    cex, returns joint output, else returns None.
    """
    # DEBUG
    global n_cex_check_calls
    n_cex_check_calls += 1
    
    x = np.copy(cex)
    
    
    for w, b in zip(weights, bias):
        x = x @ w
        x = x + b
        x[ np.where( x < 0 ) ] = 0
        
    
    return x if np.any( x @ out_lp_m > out_lp_b ) else None
    



def main(   weights : list[ArrayLike], biases : list[ArrayLike],
            pre_perm : list[int], pre_lb : ArrayLike, pre_ub : ArrayLike,
            post_perm : list[int], post_epsilon : float,
            num_cexes : Union[int, None] = None,
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
    num_cexes           -   If given, specifies the number of counterexample candidates to look for.
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
    
    # CEX scheduling
    cex_psched = PriorityPreScheduler(num_workers)


    # Set up cex candidates
    if num_cexes == None:
        num_cexes = sum(( b.shape[0] for b in biases )) * CEX_PULLBACK_NUM_TOTAL_CANDIDATES_MULT
        log("Automatically choosing to search for {0} cexes".format(num_cexes))
    n_pb_per_layer = floor((num_cexes**(1.0 / n_layers) + 1))
    log("Looking for {0} pullbacks per layer".format(n_pb_per_layer))
        
 
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
    
    # Start workers.
    log("Starting algo for {0} layers".format(n_layers))
    start()
    start_time = perf_counter()
    
    try:
        
    
        # Que up pushforward
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
        
        
        # Check if center point is a cex in itself
        ret = check_cex(inp_c, weights, biases, out_lp_m, out_lp_b)
        if ret is not None:
            log("Center point is a counterexample")
            return PermCheckReturnStruct( PermCheckReturnKind.COUNTEREXAMPLE, perf_counter()-start_time,
                    [(inp_c[:n_inputs], inp_c[n_inputs:], ret[:n_outputs], ret[post_perm],
                    ret[n_outputs:])])
        
        # Schedule pullback of out lp
        add_task( (TaskMessage.PULL_B_RELU, n_layers, out_lp_m, out_lp_b, centers[-1]) )
        
        
        # Start loop acting on returned messages
        pf_remaining = True         # Is pushforwards all done?
        pb_remaining = True         # Is pullbacks all done
        n_incl_check = 0            # Number of inclusion checks performed
        n_cex_check = 0             # Number of cex checks done
        while pf_remaining or pb_remaining or n_incl_check < n_pos or cex_psched.is_active():
          
            #log("while, {0}".format((pf_remaining, pb_remaining, n_incl_check < n_pos, n_cex_check <
            #    num_cexes, n_cex_check, num_cexes))) #DEBUG

            # Throw error if any worker has failed.
            if any_error():
                raise RuntimeError("A worker has encountered an unhandled exception")
            
            # Get returned message
            ret = get_return(wait=True)
            msg, layer, *data = ret
            log("Recieved message {0} at layer {1}".format(msg, layer)) 
          
          
            # If message says a pushforward is done, que the next if available, or set flags
            if msg == ReturnMessage.PUSH_F_LINEAR_DONE:

                # Get the postcond
                postconds[2*layer+1] = data[0]
                
                log("Layer {0} has pre-relu postcond of dim {1}".format(layer, data[0].num_neuron))
                
                # If we also have a precondtion, schedule an inclusion check
                if pre_relu_pconds[layer] is not None:
                    add_task( (TaskMessage.ICHK_PRE_RELU, layer, postconds[2*layer+1], 
                                pre_relu_pconds[layer]) )

                # Schedule next pullback
                log("Scheuling pushforward across relu layer {0}".format(layer))
                add_task( (TaskMessage.PUSH_F_RELU, layer, postconds[2*layer+1]) )
            
           
            # If message says a pushforward is done, que the next if available, or set flags
            elif msg == ReturnMessage.PUSH_F_RELU_DONE:

                # Get the postcond
                pcidx = 2 * (layer+1)           # where the next postcond will go
                postconds[pcidx] = data[0]
                
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
                    continue         # Do not stop if pullback failed, TODO branch
                    #stop()
                    #return PermCheckReturnStruct( PermCheckReturnKind.INCONCLUSIVE )
                    
                # Get the precond
                pre_relu_pconds[layer-1] = data[0]      # For now, just pick the first precond produced
                
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
                    log("Found proof via inclusion just before linear layer {0}".format(layer))
                    stop()
                    return PermCheckReturnStruct( PermCheckReturnKind.PROOF, perf_counter()-start_time,
                            postconds[:2*layer+1], pre_relu_pconds[layer:], pre_linear_pconds[layer:],
                            layer)
                
                n_incl_check += 1
                
                log("No inclusion just before linear layer {0}, found {2} cexes, checked {1} layers".format(
                            layer, n_incl_check, len(cexes)))

                # If already at layer 0, check if any cex is good
                if layer <= 0:
                    for cex in cexes:
                        log("Checking cex from inclusion in position 0")
                        ret = check_cex(cex, weights, biases, out_lp_m, out_lp_b)
                        n_cex_check += 1
                        if ret is not None:
                            log("Found cex from inclusion in position 0")
                            stop()
                            return PermCheckReturnStruct( PermCheckReturnKind.COUNTEREXAMPLE, perf_counter()-start_time,
                                    [(cex[:n_inputs], cex[n_inputs:], ret[:n_outputs], ret[post_perm],
                                    ret[n_outputs:])])
                        else:
                            log("Not a cex, {0} of {1} checked".format(n_cex_check, num_cexes))
                    
                # Schedule pullback for each returned counterexample
                else:
                    for cex in cexes:
                        log("Pre-Scheduling pullback of cex candidate across ReLU")
                        cex_psched.add_task( ( TaskMessage.CEX_PB_RELU, layer, cex,
                                                postconds[layer*2 - 1] ), 1-layer*2)
                    
                
            # If the inclusion check is done, quit if successfull, or try to pull back cex TODO cex pb
            elif msg == ReturnMessage.ICHK_PREREL_DONE:
                
                cexes = data[0]
                
                # Quit out and return if there are no cexes.
                if len(cexes) == 0:  
                    log("Found proof via inclusion just before ReLU layer {0}".format(layer))
                    stop()
                    return PermCheckReturnStruct( PermCheckReturnKind.PROOF,  perf_counter()-start_time,
                            postconds[:(layer+1)*2], pre_relu_pconds[layer:],
                            pre_linear_pconds[layer+1:], layer)
                
                n_incl_check += 1
                
                log("No inclusion at layer {0}, found {2} cexes, checked {1} layers".format(
                            layer, n_incl_check, len(cexes)))
                
                # If layer is 0, shedule linear pullbacks
                if layer <= 0:
                    for cex in cexes:
                        log("Pre-Sceduling pullback of cex from inclusion over first linear layer")
                        cex_psched.add_task(( TaskMessage.CEX_PB_LINEAR, 0, cex, postconds[0],
                                                weights[0], biases[0] ), -0)
                
                # Schedule pullback for each returned counterexample
                else:
                    for cex in cexes:
                        log("Pre-scheduling pullback of cex candidate across combined layer from {0}".format(
                                layer))
                        cex_psched.add_task(( TaskMessage.CEX_PB_LAYER, layer, cex, 
                                                postconds[layer * 2 - 1], weights[layer], 
                                                biases[layer], n_pb_per_layer ), 1 - layer*2)
            
            # If a cex has been pulled back over a combined layer, continue pullback.
            elif msg == ReturnMessage.CEX_PB_LAYER_DONE:
                
                # Let prescheduler know we have got a cex
                cex_psched.task_done()
                
                # Get cexes
                cexes = data[0]
                assert len(cexes) > 0
                
                # If next layer is 0, shedule linear pullbacks
                if layer-1 <= 0:
                    for cex in cexes:
                        log("Pre-sceduling pullback of cex from combined over first linear layer")
                        cex_psched.add_task(( TaskMessage.CEX_PB_LINEAR, 0, cex, postconds[0],
                                                weights[0], biases[0] ), -0)
                
                # Else, continue layer pullback
                else:
                    for cex in cexes:
                        log("Pre-scheduling pullback of cex candidate across combined layer from {0}".format(
                                layer-1))
                        cex_psched.add_task(( TaskMessage.CEX_PB_LAYER, layer-1, cex,
                                                postconds[layer*2 - 3], weights[layer-1],
                                                biases[layer-1], n_pb_per_layer ), 3 - 2*layer)
                    
            # If a cex has been pulled back over a linear layer, check if layer was 0, if so, check cex,
            # else, continue pullback via relu.
            elif msg == ReturnMessage.CEX_PB_LINEAR_DONE:
                
                # Let pre-scheduler know a task is done
                cex_psched.task_done()
                
                # Get cex
                cex = data[0]
                
                # If cex is none, do nothing further
                if cex is None:
                    log("No cex returned from pullback over linear layer")
                    continue
                
                # If layer is 0, check cex
                if layer <= 0:
                    log("Checking cex that has been pulled back")
                    ret = check_cex(cex, weights, biases, out_lp_m, out_lp_b)
                    n_cex_check += 1
                    if ret is not None:
                        log("Found cex that has been pulled back")
                        stop()
                        return PermCheckReturnStruct(PermCheckReturnKind.COUNTEREXAMPLE, perf_counter()-start_time,
                                [(cex[:n_inputs], cex[n_inputs:], ret[:n_outputs], ret[post_perm],
                                ret[n_outputs:])])
                    else:
                        log("Not a cex, {0} of {1} checked".format(n_cex_check, num_cexes))
                
                # Else, shchedule relu pullback. This should be unreachable code.
                else:
                    raise RuntimeError("Just did a linear cex pullback for non-zero layer")
                    
            # If a cex has been pulled back over a relu layer, schedule combined pullback, except at
            # 0 layer, where we schedule linear pullback
            elif msg == ReturnMessage.CEX_PB_RELU_DONE:
                
                
                # Let pre-scheduler know a task is done
                cex_psched.task_done()
                
                # Get cex
                cex = data[0]
                
                # If cex is none, do nothing further
                if cex is None:
                    log("No cex returned from pullback over relu layer")
                    continue
                
                # If next layer is 0, shedule linear pullbacks
                if layer-1 <= 0:
                    log("Sceduling pullback of cex from relu over first linear layer")
                    cex_psched.add_task(( TaskMessage.CEX_PB_LINEAR, 0, cex, postconds[0],
                                            weights[0], biases[0] ), -0)
                
                # Else, continue layer pullback
                else:
                    log("Scheduling pullback of cex candidate across combined layer from {0}".format(
                            layer-1))
                    cex_psched.add_task(( TaskMessage.CEX_PB_LAYER, layer-1, cex, 
                                            postconds[layer*2 - 3], weights[layer-1],
                                            biases[layer-1], n_pb_per_layer ), 3 - 2*layer)
                    

            else:
                log("Unknown return message {0}".format(msg))
      
        #DEBUG
        global n_cex_check_calls
        log("Main process coordination loop has stopped, {0}, {1}, {2}, {3}".format(
                    pf_remaining, pb_remaining, n_incl_check, n_cex_check_calls))
       
        # If we failed to find a proof or a cex, return inconclusive
        stop()
        return PermCheckReturnStruct( PermCheckReturnKind.INCONCLUSIVE, perf_counter()-start_time )
   
    except:
        stop()
        log("Unhandled exception in main process:")
        trace = format_exc()
        log(trace)
        return PermCheckReturnStruct( PermCheckReturnKind.ERROR, perf_counter()-start_time, trace )
   
   
    
if __name__ == "__main__":

    import sys
    
    # Some very simple cli
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <spec_file> <number_of_worker_processes> [output_file]")
        exit(-1)
    
    with open(sys.argv[1]) as spec:
        spec_d = eval(spec.read())
        
    ret = main( weights         = [np.array(w) for w in spec_d['weights']],
                biases          = [np.array(b) for b in spec_d['biases']],
                pre_perm        = spec_d['pre_perm'],
                pre_lb          = np.array(spec_d['pre_lb']),
                pre_ub          = np.array(spec_d['pre_ub']),
                post_perm       = spec_d['post_perm'],
                post_epsilon    = spec_d['post_epsilon'],
                num_workers     = int(sys.argv[2])
                )
    
    print("\n\n\nO U T P U T: \n\n\n")
    #print(repr(ret))
    print(str(ret))
    
    if len(sys.argv) >= 4:
        with open(sys.argv[3], 'w') as f:
            f.write(repr(ret))
        
        
    #if len(sys.argv) < 3:
    #    print("Usage: `python main.py <number_of_worker_processes>` `<unit-name>`")
    #    exit(-1)
    #    
    #if sys.argv[2] == 'safe':
    #
    #    weights = [ np.array(   [[1000, -1000, 1000, -1000],
    #                             [-1000, 1000, -1000, 1000]] ),
    #                np.array(   [[1, 0], [0, 1], [-1, 0], [0, -1]] )]
    #    biases = [ np.array( [0, 0, -1, -1] ), np.array( [0, 0] ) ]
    #    sig = [1, 0]
    #    
    #    ret = main(weights, biases, sig, np.array([0, 0]), np.array([1,1]), sig, 0.1,
    #                num_workers = int(sys.argv[1]))
    #    
    #    print("\n\nReturned {1}: {0}".format(str(ret), ret.kind.value))
    #    print("\n\nDetails:\n\n")
    #    print(repr(ret))
    #
    #elif sys.argv[2] == 'cex':
    #    
    #    weights = [ np.array(   [[1000, -1000, 1000, -1000],
    #                             [-1000, 1000, -1000, 1000]] ),
    #                np.array(   [[1, 0], [0, 1], [-1, 0], [0, -1]] )]
    #    biases = [ np.array( [0, 0, -1, -1] ), np.array( [0, 0] ) ]
    #    sigI = [1, 0]
    #    sigO = [0, 1]
    #    
    #    ret = main(weights, biases, sigI, np.array([0, 0]), np.array([1,1]), sigO, 0.1,
    #                num_workers = int(sys.argv[1]))
    #    
    #    print("\n\nReturned {1}: {0}".format(str(ret), ret.kind.value))
    #    print("\n\nDetails:\n\n")
    #    print(repr(ret))
