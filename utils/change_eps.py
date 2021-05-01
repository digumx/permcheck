"""
A short script to change the epsilon in spec files

Use: python change_eps.py <eps> <in_file> <out_file>
"""

import sys

with open(sys.argv[2], 'r') as f:
    dct = eval(f.read())
    
dct['post_epsilon'] = float(sys.argv[1])

with open(sys.argv[3], 'w') as f:
    f.write(repr(dct))
