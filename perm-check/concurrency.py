"""
Contains code for doing tasks concurrently. Uses a Server-Worker model. The Server distributes work
to many workers, sending the details via a task Queue. The workers return results via a return
Queue.
"""


from typing import Callable, Any
from enum import Enum, auto
from time import monotonic

from global_consts import USE_MP

if USE_MP:
    from multiprocessing import get_context, Queue, Event, Manager, current_process, Lock
    from queue import Empty

if USE_MP:
    from global_consts import MP_NUM_WORKER, MP_START_METHOD


if USE_MP:
    
    class ChildState:
        """
        A list of possible states a child process can be in.
        """
        EXIT    = 0             # The child has finished all work, and is ready to exit.
        IDLE    = 1             # The child is idling, waiting for more work
        BUZY    = 2             # The child is executing task
        PUSH    = 3             # The child is waiting to push results to return queue
        STRT    = 4             # The child has just been started
        
        
    def _worker(kern : Callable[..., Any], tq : Queue, rq : Queue, state : ChildState, 
                exit_ev : Event, rp : float, pl: Lock):
        """
        Keep extracting elements from tq, call kern on contents, push return to rq. Quit if the
        exit_ev has been set. Just before quitting, send a QuitMessage. Writes the current state to
        `state`.
        """
        # Set up reference point for logging
        global ref_point, print_lock
        ref_point = rp
        print_lock = pl
        
        log("Starting worker") #DEBUG
        
        has_ret = False
        ret = None
        state = ChildState.IDLE
        
        # Exit if set
        while not exit_ev.is_set():
            
            #print("Loop entry") #DEBUG
            
            if not has_ret:
                # Get a task
                try:
                    #print("Requesting task")
                    tsk = tq.get_nowait()
                    #print("Obtained task")
                    state = ChildState.BUZY
                    #print("Falg set")
                except Empty:
                    continue
                
                log("Task recieved") #DEBUG
                
                # Perform task:
                ret = kern(tsk)
                state = ChildState.PUSH
                has_ret = True
            
            # Push return
            try:
                rq.put_nowait(ret)
                state = ChildState.IDLE
            except Full:
                continue
            
            # Return has been pushed
            has_ret = False
        
        state = ChildState.EXIT
        
            

"""
SINGLE INSTANCE CLASS

Implements a pool of workers and Queues to schedule tasks and collect returns.  During construction,
a kernel function must be passed. Each worker calls the kernel function with the data passed for
each task. This also has a queue for shared timestamped logging.

Each subprocess created has an unique index. The logging process has index 0, and all the other
workers have indices from 1 onwards.


Managed globals:

    manager : Manager       -   A shared memory manager.
    task_q : Queue          -   Queue for scheduled tasks
    retn_q : Queue          -   Queue for return messages
    exit_ev : Event         -   An Event that is fired to make all workers exit as soon as possible
    states : [ChildState]   -   A list of states for each process, according to their index
    workers : [Process]     -   A list of worker Processes
    print_lock : Lock         -   A lock that must be acquired to print things


If USE_MP is False, the kernel is simply run on each task as an when it is scheduled. The
managed globals in that case are:

    retn_l : List           -   A non-shared list used to hold return values, acts as a local q.
    kernel : Callable       -   The kernel function

Common globals managed regardless of USE_MP:
    
    ref_point : float       -   Reference point for logging time
"""
if USE_MP:
    manager = None
    task_q = None
    retn_q = None
    exit_ev = None
    states = None
    workers = None
    print_lock = None

else:
    retn_l = None
    kernel = None

ref_point = None

    
def init(k : Callable[..., Any], 
                n_workers = MP_NUM_WORKER, start_method = MP_START_METHOD):
    """
    Initialize object from given data. Does nothing if mp is not enabled.
    """
    global ref_point
    ref_point = monotonic()
    
    if USE_MP:
        global manager, task_q, retn_q, exit_ev, states, workers, print_lock
        
        ctx = get_context(method=start_method)
        manager = ctx.Manager()
        task_q = ctx.Queue()
        retn_q = ctx.Queue()
        exit_ev = ctx.Event()
        print_lock = ctx.Lock()
        
        states = manager.list([ ChildState.STRT for _ in range(n_workers + 1) ])
        
        workers = [ ctx.Process(    target = _worker, 
                                    args = ( k, task_q, retn_q, states[i+1], exit_ev, ref_point,
                                                print_lock ),
                                    name = "WORKER {0}".format(i)
                                ) for i in range(n_workers) ]
        
    else:
        global retn_l, kernel
        retn_l = Queue()
        kernel = k
        

def start():
    """
    Start scheduler. Tasks scheduled will only be executed once scheduler starts. Does nothing
    if mp is not enabled.
    """
    if not USE_MP:
        return
    
    global workers
    
    for w in workers:
        w.start()
   
    while True:
        brk = True
        log(str([ s for s in states[1:] ]))
        for s in states[1:]:
            if s != ChildState.IDLE:
                brk = False
        if brk: break
   
def stop():
    """
    Force all workers to stop. All data started tasks are completed, and all scheduled tasks
    that have not been done are discarded. All data in the return queue is lost. All messages in
    the logging queue is logged. Does nothing if MP is not enabled.
    """
    if not USE_MP:
        return
    
    global exit_ev, states, task_q, retn_q, workers
    
    # Send stop message
    exit_ev.set()
    
    # Wait for all processes to be in EXIT state
    while not all(( s == ChildState.EXIT for s in states )): pass
    
    # Clear task queue.
    while True:
        try:
            task_q.get_nowait()
        except Empty:
            break
    
    # Clear return queue.
    while True:
        try:
            retn_q.get_nowait()
        except Empty:
            break
    
    # Join with all processes
    for w in workers:
        w.join()
    
    # Close all processes
    for w in workers:
        w.close()
    

def add_task(tsk):
    """
    Add a task to the task queue. If mp is disabled, just execute task and store return in
    return queue instead.
    """
    if USE_MP:
        global task_q
        
        task_q.put(tsk)
    else:
        global retn_l, kernel
        
        retn_l.append(kernel(tsk))
        
            
def get_return(wait=True):
    """
    Get a return value from one of the threads. If wait is true, blocks until there is an item
    to return, otherwise returns `None` if there is nothing to return. If mp is disabled, wait
    is ignored, and None is returned if there is nothing to return
    """
    if USE_MP:
        global retn_q
        
        try:
            return retn_q.get(block=wait)
        except Empty:
            return None
    
    else:
        global retn_l
        
        if len(retn_l) > 1:
            return retn_l.pop(-1)
    
    return None

def wait_till_all_done():
    """
    Wait until all scheduled tasks are done. Note that if another process adds tasks while this
    method is called, this method may deadlock, or return before the newly added task is
    complete. Returns immediately if mp is disabled
    """
    if not USE_MP:
        return
    
    global states
    
    while True:
        brk = True
        log(str([ s for s in states[1:] ]))
        for s in states[1:]:
            if s != ChildState.IDLE:
                brk = False
        if brk: break


def log(s : str):
    """
    Write a string to shared log. Automatically add source process and timestamp. `init()`
    must be called from server process before log is called from any process.
    """
    global ref_point 
    t = monotonic() - ref_point
    
    if USE_MP:
        global print_lock
        print_lock.acquire()
        try:
            print("[ {0} ] {1}: {2}".format(t, current_process().name, s))
        finally:
            print_lock.release()
    else:
        print("[ {0} ] {1}".format(t, s))



"""" DEBUG """""

def k(n):
    log("Squaring {0}".format(n))
    return n, n**2

if __name__ == "__main__":  #DEBUG
    
    n_inpts = 30
    
    init(k)
    log("init done")
    for i in range(n_inpts):
        add_task(i)
    log("Tasks added")
    start()
    log("Start complete")
    
    log("Getting results")
    for _ in n_inpts:
        res = get_return(wait=True)
        log(res)
        
    log("Waiting till completion")
    wait_till_all_done()
    
    log("Stopping")
    stop()