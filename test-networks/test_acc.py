"""
A script to test the accuracy of trained networks against the spec files. This prints to the output
file the percentage of input points for which the property is satisfied. Usage:

    test_acc <number of points> <output file>

Where <number of points> is the number of input points to test for
"""



import sys

import numpy as np



def test_acc_net( spec, num_points ):
    """
    Test the accuracy of one spec file using the given number of inputs. Returns accuracy as a float
    between 0 and 1.
    """
    # Get data
    weights         = [np.array(w) for w in spec_d['weights']]
    biases          = [np.array(b) for b in spec_d['biases']]
    pre_perm        = spec_d['pre_perm']
    pre_lb          = spec_d['pre_lb']
    pre_ub          = spec_d['pre_ub']
    post_perm       = spec_d['post_perm']
    post_epsilon    = spec_d['post_epsilon']
        
    # Generate input
    inp = np.stack( [ (l + np.random.rand(num_points) * (u-l)) for l, u in zip(pre_lb, pre_ub) ],
                    axis = 1 )

    # Permute input
    si_inp = inp[ :, pre_perm ]
    
    # Get outputs
    n_inp, n_si_inp = np.copy(inp), np.copy(si_inp)
    for w,b in zip(weights, biases):
        n_inp = n_inp @ w + b
        n_inp = n_inp * (n_inp > 0)
        n_si_inp = n_si_inp @ w + b
        n_si_inp = n_si_inp * (n_si_inp > 0)

    # Permute output
    so_n_inp = n_inp[ :, post_perm ]

    # Calculate accuracy
    n_correct = np.count_nonzero( np.all( np.abs(so_n_inp - n_si_inp) <= post_epsilon, axis=1 ))
    return n_correct / num_points



if __name__ == "__main__":

    files = [   './trained/trained_safe_3_eps_0.1.spec', 
                './trained/trained_safe_3_eps_0.3.spec', 
                './trained/trained_safe_3_eps_0.5.spec', 
                './trained/trained_safe_3_eps_0.7.spec', 
                './trained/trained_safe_3_eps_0.9.spec', 
                './trained/trained_safe_4_eps_0.1.spec', 
                './trained/trained_safe_4_eps_0.3.spec', 
                './trained/trained_safe_4_eps_0.5.spec', 
                './trained/trained_safe_4_eps_0.7.spec', 
                './trained/trained_safe_4_eps_0.9.spec', 
                './trained/trained_safe_5_eps_0.1.spec', 
                './trained/trained_safe_5_eps_0.3.spec', 
                './trained/trained_safe_5_eps_0.5.spec', 
                './trained/trained_safe_5_eps_0.7.spec', 
                './trained/trained_safe_5_eps_0.9.spec', 
                './trained/trained_safe_6_eps_0.1.spec', 
                './trained/trained_safe_6_eps_0.3.spec', 
                './trained/trained_safe_6_eps_0.5.spec', 
                './trained/trained_safe_6_eps_0.7.spec', 
                './trained/trained_safe_6_eps_0.9.spec', 
                './trained/trained_safe_7_eps_0.1.spec', 
                './trained/trained_safe_7_eps_0.3.spec', 
                './trained/trained_safe_7_eps_0.5.spec', 
                './trained/trained_safe_7_eps_0.7.spec', 
                './trained/trained_safe_7_eps_0.9.spec', 
                './trained/trained_safe_8_eps_0.1.spec', 
                './trained/trained_safe_8_eps_0.3.spec', 
                './trained/trained_safe_8_eps_0.5.spec', 
                './trained/trained_safe_8_eps_0.7.spec', 
                './trained/trained_safe_8_eps_0.9.spec', 
                './trained/trained_safe_9_eps_0.1.spec', 
                './trained/trained_safe_9_eps_0.3.spec', 
                './trained/trained_safe_9_eps_0.5.spec', 
                './trained/trained_safe_9_eps_0.7.spec', 
                './trained/trained_safe_9_eps_0.9.spec', 
                './trained/trained_safe_10_eps_0.1.spec', 
                './trained/trained_safe_10_eps_0.3.spec', 
                './trained/trained_safe_10_eps_0.5.spec', 
                './trained/trained_safe_10_eps_0.7.spec', 
                './trained/trained_safe_10_eps_0.9.spec', 
                ]
    
    n_points = int(sys.argv[1])
    with open(sys.argv[2], 'w') as out_file:
        out_file.write("{\n")

    for fname in files:

        with open(fname) as spec:
            spec_d = eval(spec.read())

        acc = test_acc_net(spec_d, n_points)

        print("\n{0} \t:\t {1}\n\n".format(fname, acc))
        with open(sys.argv[2], 'a') as out_file:
            out_file.write("'{0}' \t:\t '{1}',\n".format(fname, acc))

    with open(sys.argv[2], 'a') as out_file:
        out_file.write("}")
