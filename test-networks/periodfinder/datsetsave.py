"""
Command line usage:
    arg1    -   Number of samples per datapoint
    arg2    -   Number of data points
    arg3    -   Sparsity
    arg4    -   Output filename
"""

from datasetgen import datasetgen
import sys

sp=int(sys.argv[1])
dp=int(sys.argv[2])
s=int(sys.argv[3])
dataset_filename=sys.argv[4]
l=str(datasetgen(sp,dp,s))

text_file = open(dataset_filename,"w")

n=text_file.write(l)
text_file.close()
