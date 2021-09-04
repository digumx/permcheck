"""
A script to train neural networks close to max_n using a bunch of simple examples. This uses the
same architecture as the max_n nets, but decides the weights via training. To train, a number of
random samples are generated with the guarantee that no two distinct sapmles have components closer
than 1/`granularity`

Command line args:
    
    1.  Number of inputs
    2.  Granularity of training data
    3.  Size of training set
    4.  Batch size for training
    5.  Number of epochs to train for
    6.  Weather to use 'safe' or 'unsafe' output permutation
    7.  The tolerance for output permutations
    8.  Output spec file


"""


import sys

from numpy import array, newaxis, argmax, zeros
from numpy.random import randint
from torch import full, as_tensor, transpose
from torch.optim import SGD, Adam
from torch.nn import Module, Linear, ReLU, MSELoss, init
from torch.utils.data import TensorDataset, DataLoader


class MaxN(Module):
    """
    A nnet with n inputs and n outputs with the following approximate trained behavior: If the i-th
    input is max, the i-th output is one, and the rest are 0.
    """
    
    def __init__(self, num_inp : int):
        """
        Make a net with `num_inp` inputs
        """
        super(MaxN, self).__init__()
        
        # Sanity checks
        assert num_inp >= 3
        
        # Set up layers
        self.n = num_inp
        self.n2 = num_inp * (num_inp - 1)
        self.n1 = self.n2 * 2
        self.l1 = Linear(self.n, self.n1)
        self.r1 = ReLU()
        self.l2 = Linear(self.n1, self.n2)
        self.r2 = ReLU()
        self.l3 = Linear(self.n2, self.n)
        self.r3 = ReLU()
        
        # Init all weights and biases to 0.
        #self.l1.weights.fill_(0)
        #self.l2.weights.fill_(0)
        #self.l3.weights.fill_(0)
        #self.l1.bias.fill_(0)
        #self.l2.bias.fill_(0)
        #self.l3.bias.fill_(0)
        
        # Initialize with glorot
        init.xavier_normal(self.l1.weight)
        init.xavier_normal(self.l2.weight)
        init.xavier_normal(self.l3.weight)
        init.uniform(self.l1.bias)
        init.uniform(self.l2.bias)
        init.uniform(self.l3.bias)
        
    
    def forward(self, x):
        """
        Convert input `x` to output by passing through network
        """
        return self.r3( self.l3( self.r2( self.l2( self.r1( self.l1( x ))))))

    def write_spec(self, filename, safe, eps):
        """
        Write to given spec file. Safe should be a string value saying 'safe' or 'unsafe'. The
        tolerance on the output side is set by `eps`.

        """
        
        # Get dict
        dct = {}
        dct['weights'] = [  transpose(self.l1.weight.data, 0, 1).tolist(),
                            transpose(self.l2.weight.data, 0, 1).tolist(),
                            transpose(self.l3.weight.data, 0, 1).tolist() ]
        dct['biases'] = [   self.l1.bias.tolist(),
                            self.l2.bias.tolist(),
                            self.l3.bias.tolist() ]
        dct['pre_lb'] = [0] * self.n
        dct['pre_ub'] = [1] * self.n
        dct['pre_perm'] = [ i for i in range(1, self.n) ] + [0]
        if safe == 'safe':
            dct['post_perm'] = [ i for i in range(1, self.n) ] + [0]
        elif safe == 'unsafe':
            dct['post_perm'] = list(range(self.n))
        else:
            raise ValueError("Unknown string for safe")
        dct['post_epsilon'] = eps
        
        with open(filename, 'w') as f:
            f.write(repr(dct))


if __name__ == "__main__":
    
    # Get params
    n           = int(sys.argv[1])
    gran        = int(sys.argv[2])
    data_size   = int(sys.argv[3])
    batch_size  = int(sys.argv[4])
    n_epochs    = int(sys.argv[5])
    
    # Generate data
    print("Generating random data")
    data = randint(0, gran, size=(data_size, n))
    data /= full((data_size, n), gran)
    
    # Generate correct outputs
    print("Generating correct ouput")
    maxidx = argmax(data, axis=1)
    out = zeros((data_size, n))
    print(out.shape)
    print(maxidx.shape)
    out[range(data_size), maxidx] = 1

    #DEBUG
    print("Data:")
    print(data[:10])
    print(out[:10])
    print(maxidx[:10])

    # Make a DataLoader
    print("Making a loader")
    dset = TensorDataset(as_tensor(data), as_tensor(out))
    dloader = DataLoader(dset, batch_size)
    
    # Get the model, loss and optimizer
    print("Setting up nnet, loss and optimizer")
    nnet = MaxN(n)
    mse_loss = MSELoss()
    opt = Adam(nnet.parameters())
    
    # Start the training process
    try:
        for epoch in range(n_epochs):
            print(f"Epoch {epoch} of {n_epochs}")

            for batch, (x, y) in enumerate(dloader):
                
                # Calculate loss
                p = nnet(x)
                l = mse_loss(p, y)
                
                # Backprop grads
                opt.zero_grad()
                l.backward()
                opt.step()
                
                print(f"Loss {l.item()} at {batch * batch_size} of {data_size}        ", end='\r') 

            print()

    except KeyboardInterrupt:
        print()
        print("Stopping training")

    
    print("Writing to file")
    nnet.write_spec(sys.argv[8], sys.argv[6], float(sys.argv[7]))
