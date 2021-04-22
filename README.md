A very basic and potentially inefficient implementation of the algorithm.

# Dependencies:

The program relies on python typechecking, and thus needs python 3.8 or newer to work reliably.
Additionally, the following python libraries are needed to run the code:

 -  numpy
 -  scipy

# Running:

Currently, the `main.py` file can be run with a fixed simple example. To run the code, use the
command from within the `perm_check` directory:

```
python main.py <num_workers>
```

Where `num_workers` is the number of child worker processes that are spawned to run the code. Must
be atleast `1`. Recommended: number of cores in system - 2.
