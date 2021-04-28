"""
A simple script to generate simple safe and unsafe examples to try the code on. For a given n, this
generates a DNN that has n inputs and n outputs, with the approximate behavior that the i-th output
is 1 only if the ith input is maximum, else it is 0. Then, it generates a spec file that checks if
cyclically permuting the inputs by 1 permutes the outputs cyclically by 1 as well, for the safe
case, and weahter the outputs remain unpermuted for the unsafe case. In iether case, an epsilon of
0.1 is used, and the inputs are bounded between 0 and 1.

Usage:
    python generate.py <safe | unsafe> <n> <out_file>
    
"""


import sys
from itertools import permutations


# Constants
sharpness = 1000    # Increase this to reduce the approximateness of the calcuclations


if __name__ == "__main__":
    
    # Basic checks
    if len(sys.argv) < 4:
        print("Usage:   python generate.py <safe | unsafe> <n> <out_file>")
    n = int(sys.argv[2])
    assert n >= 2
    
    # List of pairs of inputs
    prs = list(permutations(range(n), 2))
    
    # Layer 1 computes each pairwise difference between each input times sharpness twice, once with
    # -1 added, and again with no -1 added.
    n1 = 2 * n * (n - 1)
    w1 = [ [ sharpness if i == prs[j][0] else ( -sharpness if i == prs[j][1] else 0 ) 
                    for j in range(n * (n-1)) ] * 2 for i in range(n) ]
    b1 = [0] * (n * (n-1)) + [-1] * (n * (n-1))
    
    # Layer 2 subtracts the two versions of difference calculated for each difference. Each point
    # here is 1 if a pair comparision of the inputs holds, else it is 0.
    n2 = n * (n - 1)
    w2 = [ [ 1 if i == j else 0 for j in range(n2) ] for i in range(n2) ] + \
         [ [-1 if i == j else 0 for j in range(n2) ] for i in range(n2) ]
    b2 = [ 0 for _ in range(n2) ]
    
    # Layer 3 performs a logical and operation of i < j over all j for each i.
    w3 = [ [ 2 if j == prs[i][0] else 0 for j in range(n) ] for i in range(n2) ]
    b3 = [ 3 - 2*n for _ in range(n) ]
    
    # Get the perms
    pre_perm = [ i + 1 for i in range(n-1) ] + [0]
    if sys.argv[1] == "safe":
        post_perm = pre_perm[:]
    else:
        post_perm = [ i for i in range(n) ]
        
    # Get bounds
    ub = [ 1 for _ in range(n) ]
    lb = [ 0 for _ in range(n) ]
    
    # Compose the dict
    dct = { 'weights' : [w1, w2, w3], 'biases' : [b1, b2, b3], 'pre_perm' : pre_perm, 
            'post_perm' : post_perm, 'pre_lb' : lb, 'pre_ub' : ub, 'post_epsilon' : 0.1 }
    
    # Write the object to given file
    with open(sys.argv[3], 'w') as f:
        f.write(repr(dct))
