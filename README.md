A very basic and potentially inefficient implementation of the algorithm.

# Dependencies:

The program relies on python type hints, and thus needs python 3.8 or newer to work reliably.
Additionally, the following python libraries are needed to run the code:

 -  numpy
 -  scipy

# Running:

To run the code, use the following command:

```
python main.py <spec_file> <num_workers>
```

Where `spec_file` is a file with all the details of the network, precondition and postcondition,

And `num_workers` is the number of child worker processes that are spawned to run the code. Must
be at-least `1`. Recommended: number of cores in system - 1.

# Spec File Format:

The spec file should contain a single python style dict with the following keys:

    weights         -   A list of weights, one for each layer, each of which is a matrix given by a
                        list of list of floats
    biases          -   A list of corresponding bias vectors
    pre_perm        -   The permutation on the input side. This should be a list of ints, the i-th
                        int being the index of the component the i-th component is sent to under the
                        permutation
    pre_lb          -   A list of lower bounds for each component of the input vector
    pre_ub          -   A list of upper bounds for each component of the input vector
    post_perm       -   The permutation on the output side. This should be a list of ints, the i-th
                        int being the index of the component the i-th component is sent to under the
                        permutation
    post_epsilon    -   The tolerance allowed for the output to satisfy the output permutation.
