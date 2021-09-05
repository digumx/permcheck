"""
A file to run all the benchmarks on perm check. Outputs the resutls and time taken for each file in
`bench_list` to given output file.

Usage:
    
    run_benchmarks.py <output_file>

"""

import sys

import numpy as np

import main



bench_list = [  '../test-networks/max_n/trained_safe_3_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_3_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_3_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_3_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_3_eps_0.9.spec', 
                '../test-networks/max_n/trained_safe_5_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_5_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_5_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_5_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_5_eps_0.9.spec', 
                '../test-networks/max_n/trained_safe_6_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_6_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_6_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_6_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_6_eps_0.9.spec', 
                '../test-networks/max_n/trained_safe_7_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_7_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_7_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_7_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_7_eps_0.9.spec', 
                '../test-networks/max_n/trained_safe_8_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_8_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_8_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_8_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_8_eps_0.9.spec', 
                '../test-networks/max_n/trained_safe_9_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_9_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_9_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_9_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_9_eps_0.9.spec', 
                '../test-networks/max_n/trained_safe_10_eps_0.1.spec', 
                '../test-networks/max_n/trained_safe_10_eps_0.3.spec', 
                '../test-networks/max_n/trained_safe_10_eps_0.5.spec', 
                '../test-networks/max_n/trained_safe_10_eps_0.7.spec', 
                '../test-networks/max_n/trained_safe_10_eps_0.9.spec', 
                ]

with open(sys.argv[1], 'w') as out_file:
    out_file.write("{\n")

for fname in bench_list:
    
    print("\n{0}\n".format(fname))

    with open(fname) as spec:
        spec_d = eval(spec.read())
        
    ret = main.main(weights         = [np.array(w) for w in spec_d['weights']],
                    biases          = [np.array(b) for b in spec_d['biases']],
                    pre_perm        = spec_d['pre_perm'],
                    pre_lb          = np.array(spec_d['pre_lb']),
                    pre_ub          = np.array(spec_d['pre_ub']),
                    post_perm       = spec_d['post_perm'],
                    post_epsilon    = spec_d['post_epsilon'],
                    num_workers     = 1
                    )
    
    print("\n{0} \t:\t {1}\n\n".format(fname, str(ret)))
    with open(sys.argv[1], 'a') as out_file:
        out_file.write("'{0}' \t\t:\t '{1}',\n".format(fname, str(ret)))

with open(sys.argv[1], 'a') as out_file:
    out_file.write("}")
