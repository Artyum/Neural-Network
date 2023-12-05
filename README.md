# Machine Learning according to Andrew Ng's Tutorial

This repository contains Python code developed in accordance with Andrew Ng's tutorial series on YouTube, available on the "Artificial Intelligence - All in One" channel. The first episode of the series can be found [here](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=1). The filenames correspond to the lecture numbers.

# Neural Network

This package includes a Python implementation of a feed-forward Neural Network, created based on Andrew Ng's tutorial. A few features have been added for learning purposes.

[Lecture 9.2 — Neural Networks Learning | Backpropagation Algorithm](https://www.youtube.com/watch?v=x_Eamf8MHwU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=51)<br/>
[Lecture 9.3 — Neural Networks Learning | Backpropagation Intuition](https://www.youtube.com/watch?v=mOmkv5SI9hU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=52)

This implementation includes:
* Logistic cost function,
* Logistic cost function,
* Sigmoid activation function,
* Backpropagation,
* Calculation of the bias gradient,
* Regularization,
* Dropout,
* Splitting the dataset into mini-batches,

You can also:
* Create any number of layers with any number of perceptrons,
* Import PNG files and predict self-painted samples (MNIST format 28x28px),
* Save and load weights,
* Evaluate the model,
* Print weights,
* Plot training progress,
* Automatically save weights at the best score,
* Automatically stop training after a predefined number of epochs (two parameters: max_epoch, patience),

This Neural Network can be used for any classification problem. The example implementation comes with the MNIST dataset, but you can feed in any other dataset you like.
To get it working, you will need [mnist-in-csv](https://www.kaggle.com/oddrationale/mnist-in-csv).

Additionally, I've uploaded some custom-made samples for testing. You can create your own samples in any graphical application like MS Paint.

The data structure used by the functions is very close to that described in Andrew Ng's tutorial, with a few exceptions:

ANgT:
* The input layer is counted as the first layer,
* Weights and biases of layer 'i' connect layers 'i' and 'i+1'.

This implementation:
* The input layer is 'virtual' and is not considered a layer in the code. It is rather an input vector of features. The first layer (with index 0) is the first hidden layer,
* Weights and biases of layer 'i' connect layer 'i-1' and 'i'. For the first layer (index 0), the weights are multiplied by input features.

The input vector of one sample has the shape (784,1) - one column of X features (28*28 for the MNIST image). The whole training dataset shape is (784,60000), etc.
The output vector is a 1-column array for each test sample.

Example of a NeuralNetwork model with a description of features:

```python
NeuralNetwork(
  input_dim=784,          # Input layer size
  max_epoch=1000,         # Stop training after a total number of epochs
  train_rate=2.0,         # Learning rate
  accuracy=1e-6,          # Stop training if the cost function change is less than accuracy
  regularization=0.0001,  # Regularization rate / 0 - disable regularization
  batch_size=256,         # Mini-batch size
  patience=20,            # Stop after a number of epoch countdowns since the best score
  dropout=0.1,            # Drop % of weights in every layer / 0 - disable dropout
  drop_epoch=15,          # Dropout every 'drop_epoch' number since the best score
  plot=False,             # Enable plot of cost and score after training
  verbose=False,          # Show messages (saving weights | dropout)
  file_prefix='mnist'     # File name prefix for weight save and load / None to disable saving weights
)
```

New layers can be added by invoking add_layer method:
```python
network.add_layer(units=700, epsilon=0.7)
network.add_layer(units=600, epsilon=0.6)
network.add_layer(units=500, epsilon=0.5)
```
Last added layer is output layer. For MNIST dataset it contains 10 perceptrons (10 classes)
```python
network.add_layer(units=10)
```

The example evaluation of this Neural Network implementation on mnist test dataset (10'000 samples):

Model: 784->800->10
Accuracy: 0.9844 (9844)
Loss: 0.0157 (156)

Model: 784->300->300->300->10
Accuracy: 0.9842 (9842)
Loss: 0.0158 (158)

Pretrained weights included for network 784->800->10:
mnist_784_800_10_l0.npy
mnist_784_800_10_l1.npy


I strive to align the nomenclature closely with that used in the Keras framework to minimize confusion.

If you have any suggestions or encounter a bug, please let me know.

Many thanks to Mr. Andrew Ng for his enlightening tutorial.
