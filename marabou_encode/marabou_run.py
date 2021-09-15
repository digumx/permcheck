"""
Given a spec file, encode and run Marabou to verify it

usage python marabou_run.py <spec_file>
"""



import sys

from maraboupy import Marabou
from maraboupy import MarabouCore


def run_marabou_on_spec( spec_fname : str, to_seconds : int = 120 ) -> tuple[bool, float, int, bool]:
    """
    Check the specification located at the passed filename using Marabou. Arguments are:
    
    spec_fname  -   The filename of the specification file
    to_seconds  -   The timeout in seconds, default is 120. Must be int.
    
    Prints results, and
    returns:
    1.  Bool, true if timeout happened
    2.  Total time taken in seconds
    3.  The number of splits.
    4.  The result as a bool, true if the spec is proved to be valid, false if cex is found, None if
        TO occurred.
    """
    
    # Read in the file
    with open(spec_fname, 'r') as f:
        in_dict = eval(f.read())
    weights = in_dict['weights']
    biases = in_dict['biases']
    pre_lb = in_dict['pre_lb']
    pre_ub = in_dict['pre_ub']
    pre_perm = in_dict['pre_perm']
    post_perm = in_dict['post_perm']
    post_epsilon = in_dict['post_epsilon']

    # Get number of variables
    n_inps = len(weights[0])
    n_vars = (n_inps + sum(( len(b) for b in biases )) * 2) * 2
    out_off = (n_inps + sum(( len(b) for b in biases[:-1] ))) * 2

    # DEBUG
    print("Number of vars {0}".format(n_vars))

    # Init maraboucore
    iq = MarabouCore.InputQuery()
    iq.setNumberOfVariables(n_vars)

    # Encode DNN. Variables are indexed as: position * 2 for primed, position * 2 + 1 for unprimed
    off = 0
    print("Encoding DNN")
    for w, b in zip(weights, biases):
        
        # Build the linear layer
        for out_i in range(len(b)):
            
            # Build the equations
            eqn = MarabouCore.Equation()
            eqn_ = MarabouCore.Equation()
            
            # Add the weights
            for in_i in range(len(w)):
                eqn.addAddend(w[in_i][out_i], (off + in_i) * 2)
                eqn_.addAddend(w[in_i][out_i], (off + in_i) * 2 + 1)

            # Equate with output var
            eqn.addAddend(-1, (off + len(w) + out_i) * 2)
            eqn_.addAddend(-1, (off + len(w) + out_i) * 2 + 1)

            # Add bias
            eqn.setScalar(-b[out_i])
            eqn_.setScalar(-b[out_i])
        
            # Add the equations
            iq.addEquation(eqn)
            iq.addEquation(eqn_)
        
        # Input vars are now output of previous layer
        off += len(w)
        
        # Build relu layer
        for i in range(len(b)):
            MarabouCore.addReluConstraint(iq, (off + i) * 2, (off + len(b) + i) * 2)
            MarabouCore.addReluConstraint(iq, (off + i) * 2 + 1, (off + len(b) + i) * 2 + 1)

        # Set offset for next layer
        off += len(b)

    # DEBUG
    print("Last offset {0}".format(off))


    # Encode input bounds
    print("Encoding bounds")
    for i, (l, u) in enumerate(zip(pre_lb, pre_ub)):
        iq.setLowerBound(i*2, l)
        iq.setLowerBound(i*2 + 1, l)
        iq.setUpperBound(i*2, u)
        iq.setUpperBound(i*2 + 1, u)
        
    # Encode input permutation
    print("Encoding input side permutation")
    for i, si in enumerate(pre_perm):
        eqn = MarabouCore.Equation()
        eqn.addAddend(1, 2*i)
        eqn.addAddend(-1, 2*si + 1)
        eqn.setScalar(0)
        iq.addEquation(eqn)
        
    # Encode output condition as disjunction
    print("Encoding output condition")
    disj = []
    for i, si in enumerate(post_perm):
        
        # Build inequalities
        eqn = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)     # For x - y >= eps
        eqn_ = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)    # For y - x >= eps
        eqn.addAddend(1, (off + i)*2)
        eqn.addAddend(-1, (off + si)*2 + 1)
        eqn_.addAddend(-1, (off + i)*2)
        eqn_.addAddend(1, (off + si)*2 + 1)
        eqn.setScalar(post_epsilon)
        eqn_.setScalar(post_epsilon)
        
        # Add ineqs
        disj.append([eqn])
        disj.append([eqn_])

    # Add the disjunction
    MarabouCore.addDisjunctionConstraint(iq, disj)
       
       
    # Solve
    options = Marabou.createOptions(timeoutInSeconds = to_seconds)
    cex, stats = MarabouCore.solve(iq, options, "")
    tm = stats.getTotalTime() / 1000
    splt = stats.getNumSplits()

    # Print basic stats and return
    print("\nDone in {0} seconds with {1} splits, ".format(tm, splt), end='')
    if stats.hasTimedOut():
        print("TO occurred")
        return True, tm, splt, None
    elif len(cex) > 0:
        print("CEX found:")
        for i in range(n_inps):
            print("{0}, ".format(cex[2*i]))
        return False, tm, splt, False
    else:
        print("No CEX found")
        return False, tm, splt, True
    
    
if __name__ == "__main__":
    run_marabou_on_spec(sys.argv[1])

