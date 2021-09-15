A prototype implementation of the algorithm using python, numpy and scipy.

# Dependencies:

The dependencies of this project are as follows:

 -  python (minimum version 3.8)
 -  numpy
 -  scipy
 -  Marabou (Only for running the Marabou comparisions)

# Installation and Running:

Once the above dependencies are installed, no further steps are necessary. PermCheck can then be
run in two ways: via a standalone script that checks a single specification, or from a very simple
python interface.

## Running Standalone for a Single Specification:

To check a single specification, it must be given as a `.spec` file described below. For such a
file, PermCheck can be run via the `run_single_specification.sh` script as follows:

```
./run_single_specification.sh <spec_file> <num_workers> [output_file]
```

Where `spec_file` is a path to the `.spec` file, `num_workers` represents the number of parallel
workers to use, and `output_file` is an optional file to which the proof or counterexample found is
written. NOTE: Currently, the code is optimized to run with one worker. This call produces a log
file in the `logs` folder with a name corresponding to the time the call was made. The contents of
this file is also printed out to standard input. The `.spec` file format is described below:

### Spec File Format:

A `.spec` file contains all the information defining a given specification. This includes the DNN,
input and output permutation, bounds for the input, and tolerance on the output side. So that the
file is easily readable both by a human and from a python script, we use a file that contains a
single python style dict. This dict should have the following keys:

 -  `weights`:            A list of weights, one for each layer, each of which is a matrix given by a
                          list of list of floats
 -  `biases`:             A list of corresponding bias vectors
 -  `pre_perm`:           The permutation on the input side. This should be a list of ints, the i-th
                          int being the index of the component the i-th component is sent to under the
                          permutation
 -  `pre_lb`:             A list of lower bounds for each component of the input vector
 -  `pre_ub`:             A list of upper bounds for each component of the input vector
 -  `post_perm`:          The permutation on the output side. This should be a list of ints, the i-th
                          int being the index of the component the i-th component is sent to under the
                          permutation
 -  `post_epsilon`:       The tolerance allowed for the output to satisfy the output permutation.

## Running via the Python Interface:

Currently, the python interface consists of a single function `perm_check.main.main`. To use this
function, import it and call it with the arguments:

 -  `weights, biases`:      A list of the weights and biases of the DNN. If the weights take n
                            neurons to m neurons, the weight matrix should have shape (n, m)
 -  `pre_perm, post_perm`:  The pre and post permutations. Permutations are given as lists of
 -                          integers, so that the i-th integer is the position where i is mapped to
 -  `pre_ub, pre_lb`:       The upper and lower bounds within which the input may lie, both are
                            ArrayLikes
 -  `post_epsilon`:         The tolerance for being the output being within the permutation
 -  `num_cexes`:            If given, specifies the number of counterexample candidates to look for.
 -  `num_workers`:          The number of worker processes to pass

# Running the Experiments:

There are several example `.spec` files that have been provided. There are two kinds of examples:
hand-crafted ones provided in `test-networks/hand-crafted`, and trained ones provided in
`test-networks/trained`. We have provided a script to run the all the experiments on these examples
with a single command:

```
./run_experiments.sh
```

This will produce the following files:

 -  `accuracy.log`:                             Lists the measured accuracy of all the trained
                                                neural networks
 -  `perm_check_results.log`:                   Lists the results of running PermCheck on the
                                                examples
 -  `marabou_results.log`:                      Lists the results of running Marabou on the examples
 -  `logs/perm_check_log_<date-time>.log`:      Combined log file for all the PermCheck runs
 -  `logs/marabou_log_<date-time>.log`:         Combined log file for all the Marabou runs

Alternatively, one can run each part of the experiments individually:

## Testing Accuracy of the Trained Examples:

This can be done via the script `test-networks/test_acc.py`. Usage:

```
test-networks/test_acc.py <num_points> <out_file>
```

This will test all the networks with `num_points` input points, and store the result in `out_file`.

## Running Marabou on the Examples:

All the marabou examples can be run with one command by running the `run_marabou_experiments.py`
script with an output file:

```
python run_marabou_experiments.py <output_file>
```

Alternatively, to check a single `.spec` file using Marabou, the `marabou_encode/marabou_run.py`
script can be used:

```
python marabou_run.py <spec_file>
```

## Running PermCheck on the Examples:

PermCheck can be run on all the examples at once via the following:

```
python run_perm_check_experiments.py <output_file>
```

To run PermCheck on a single example, refer to the previous sections.

## Generating the Examples:

To generate the hand-crafted example, the `test-networks/hand-crafted/generate.py` script can be used:

```
python generate.py <safe | unsafe> <number_of_inputs> <output_spec_file>
```

This produces safe or unsafe examples with given input size, as requested. In the `.spec` files
produced, the value of epsilon is 0.1. To use a different value, first generate the file using the
`generate.py` and then change it's epsilon using `utils/change_eps.py`, described in a later
section.

To train an example, the `test-networks/trained/train_max_n.py` script is used:

```
python train_max_n.py <number_of_inputs> <granularity> <train_size> <batch_size> <num_epochs> <safe|unsafe> <tolerance> <out_file>
```

 - `number_of_inputs`:  Number of inputs.
 - `granularity`:       Granularity of training data. Higher this value, smaller the distance
                        between the training points produced
 - `train_size`:        Size of training set
 - `batch_size`:        Batch size for SGD
 - `num_epochs`:        Number of epochs to train for
 - `safe|unsafe`:       Weather to use 'safe' or 'unsafe' output permutation
 - `tolerance`:         The tolerance for output permutations
 - `out_file`:          Output `.spec` file

## Changing Epsilon for a Generated `.spec` File:

For a `.spec` file that has already been generated, `utils/change_eps.py` can be used to change the
value of epsilon present in the file:

```
python change_eps.py <epsilon> <input_spce_file> <output_spec_file>
```

# Known Issues:

There are a few minor known issues with this current implementation:

 -  The current implementation is optimized to work with only one worker.
 -  Due to the way python handles multiprocess concurrency, the code sometimes hangs just before exit.
    Note that at this stage, the execution is complete, the timer has stopped, and all output files
    have already been written. Thus, the result can still be inspected, and the output time
    remains valid
