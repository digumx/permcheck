The code for this example was refactored from the assignment 2 submission. The dataset and
classification problem is exactly the same as that in assignment 2.

## List of source files:

There are four source files in the directory. They are listed and described below:

- `game_props.py` contains several methods for encoding various facts about the game into z3. Note
  that two conventions are used when writing functions encoding properties in z3, one takes some z3
  expressions and produces one or more z3 expression capturing the property we would like to encode.
  In some cases, the resulting expressions may be of boolean type and may capture certain conditions
  that will be later combined with other conditions and asserted. The other convention is one where
  the function is passed input and output expressions as well as a solver, and it encodes the
  required relation between input and output expressions by adding constraints to the solver.

- `datagen.py` contains a single function that generates data. This function also caches the data in
  a file `./out/data.val`, and in subsequent calls it reads this file to get parts of the requested
  data. Data read from the cache file is shuffled before returning, and if the length of data
  requested is more than that present in the file, it generates the excess.

- `train_net.py` is the main runnable script. It calls the functions to generate the data, uses
  tensorflow to train the network, and then uses z3 and the various helper encoding functions to
  verify or generate counterexamples for the properties stated in the problem. It accepts command
  line arguments for data size, training epoch length and training-testing data split. The default
  values for these arguments achieves the desired accuracy for the neural network. The neural
  network has 2 hidden layers with 20 and 10 neurons respectively. Note that the network is saved as
  `./out/model`, and in subsequent calls it is reloaded from there. To retrain the network, delete
  the model file.

## Running the code:

To run the code, simply run the `train_net.py` python script. It optionally accepts upto three
positional command line arguments. The first is the number of data points to generate, and defaults
to 10000. The second is the number of epochs to train for, and defaults to 300. The third is the
ratio of training data to total data, the rest of the data is testing data. It defaults to 0.6.
Running without any arguments uses all the default values and produced a network that met the
required accuracies. Although more time consuming, running it with the following command produces a
very accurate network, which nonetheless fails the verification:

```
python train_net.py 20000 600
```
