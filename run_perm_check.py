"""
A script that gives a very simple cli to check a .spec file using PermCheck. Usage:

python main.py <spec_file> <number_of_worker_processes> [output_file]
"""



import sys
import numpy as np
from perm_check import main



if len(sys.argv) < 3:
    print("Usage: python main.py <spec_file> <number_of_worker_processes> [output_file]")
    exit(-1)

with open(sys.argv[1]) as spec:
    spec_d = eval(spec.read())
    
ret = main.main(weights         = [np.array(w) for w in spec_d['weights']],
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
    
    
