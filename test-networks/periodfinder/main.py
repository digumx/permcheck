# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:17:57 2020

@author: sgtbh

Command line usage:
    arg1    -   Model file name. If provided and a model exists at the given location, it
                loads the model, in which case all other arguments except the second one are
                ignored. If no model exists in the given location, it saves the model.
    arg2    -   How much to shift the wave if the permutation
    arg3    -   Dataset file name
    arg4    -   Number of sample points in the dataset
    arg5    -   Number of layers
    arg6    -   Batch size, set between 50-100 for best results
    arg7    -   Number of epochs
    arg8    -   Sparsity of the data
"""
from training import dnn
import sys
import os
from tensorflow.keras import models, layers

model_file = sys.argv[1]

if os.path.isdir(model_file):
    print("Loading model")
    mdl = models.load_model(model_file)
else:
    print("Creating and training model")
    dataset=sys.argv[3]
    sample_point=int(sys.argv[4])
    num_layer=int(sys.argv[5])
    batch_size=int(sys.argv[6])
    epoch=int(sys.argv[7])
    sparsity=int(sys.argv[8])


    mdl = dnn(dataset,sample_point,num_layer,epoch,batch_size,sparsity)
    mdl.save(model_file)


# This is so that if run as a script, we can import siblings and ancestors
if __name__ == "__main__":
    import sys
    sys.path.insert(1, "../..")
from direct_check import *
from perm_check import *
import matplotlib.pyplot as plt

# We get the weights and biases of the network
weights = [l.get_weights()[0].tolist() for l in mdl.layers]
biases =  [l.get_weights()[1].tolist() for l in mdl.layers]

inp_size = len(weights[0])

shift = int(sys.argv[2])
perm = [ (i+shift)%inp_size for i in range(inp_size) ]

# Linconds for period perd
perd = 100
m = inp_size//perd
lin_conds = [ [ 0 for _ in range(inp_size + 1) ] for _ in range(inp_size - perd)]
for i in range(perd):
    for j in range(m-1):
        lin_conds[j*perd + i][j*perd + i] = 1
        lin_conds[j*perd + i][(m-1)*perd + i] = -1

print("Verifying")
res, mdl = perm_check(weights, biases, perm, lin_conds)
if res:
    print('Verified')
else:
    print('Not verified') #, model: ', mdl)
    plt.plot([ i/inp_size for i in range(inp_size) ], mdl)
    plt.show()
