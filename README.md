# HMM-LFM
Hidden Markov Models (HMMs) with Latent Force Models (LFMs) as emission process.

An implementation of the model described in [1]. Moreover, all the scripts used to run the experiments reported in [1] are available for reproducibility.

## Dependencies ##
* NumPy.
* SciPy.
* [GPy](https://github.com/SheffieldML/GPy). We currently rely on GPy to get the MOCAP dataset transparently in NumPy arrays.

## Installing ##
TODO

## Running ##

See the WalkingExperiment directory and run experiment.py (to get the results over MOCAP data in [1]) or run synthetic_experiment.py to get the toy results reported in the same paper.

Notice that the hmm directory will need to be added to your Python path if it isn't already discoverable by the Python interpreter.

## References ##
1. D. Agudelo-España, M. A. Álvarez, and Á. A. Orozco, [Definition and Composition of Motor Primitives Using Latent Force Models and Hidden Markov Models](http://dx.doi.org/10.1007/978-3-319-52277-7_31). Cham: Springer International Publishing, 2017, pp. 249–256. [Online].
