# Backpropagation-Algorithm-Neural-Networks
Implementing the backpropagation algorithm for Neural Networks

This python program implements the backpropagation algorithm for Neural Networks.
There are two steps:
1. Pre-processing the dataset. The two arguments for the program:
  - input path of the raw dataset
  - output path of the pre-processed dataset
2. Training a Neural Network - Uses the processed dataset to build a neural network. The input parameters to the neural net are:
  - input dataset – complete path of the post-processed input dataset
  - training percent – percentage of the dataset to be used for training
  - maximum_iterations – Maximum number of iterations that the algorithm will run. This parameter is used so that the program terminates       in a reasonable time.
  - number of hidden layers
  - number of neurons in each hidden layer
  
Pandas is used for reading/pre-processing data.
