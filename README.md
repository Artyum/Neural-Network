This is a Feed-forward Neural Network implementation in Python. It was created according to Andrew Ng's Tutorial (ANgT) on YouTube. The series is available on "Artificial Intelligence - All in One" channel.

This implementation includes:
* Logistic Cost Function,
* Backpropagation,
* Bias calculation,
* Regularization,
* Dropout

You can:
* Create any number of layers with any number of perceptrons
* Import png files and predict self-painted samples
* Save and load weights
* Evaluate model
* Print weights

The data structure used by functions is very close to described in A.Ng's tutorial with exceptions:

ANgT:
* The input layer is counted as first layer,
* Weights and biases of layer 'i' connects layers 'i' and 'i+1'

This implementation:
* The input layer is 'virtual' and is not considered as layer in code. It is rather a input vector of features. First layer (with index 0) is the first hidden layer,
* Weights biases of layer 'i' connects layer 'i-1' and 'i'. For the first layer (index 0) the weights are multiplied by input features.

Example of NeuralNetwork model with description of features:

```python
NeuralNetwork(
  input_dim=784,          # Input layer size
  max_epoch=1000,         # Stop training after total number of epochs
  train_rate=2.0,         # Learning rate
  accuracy=1e-6,          # Stop training if cost function change is less than accuracy
  regularization=0.0001,  # Regularization rate / 0 - disable regularization
  batch_size=256,         # Mini batch size
  patience=20,            # Stop after number of epochs countdown since best score
  dropout=0.1,            # Dropps % of weights in every layer / 0 - disable dropout
  drop_epoch=15,          # Dropout every 'drop_epoch' number since best score
  plot=False,             # Enable plot of cost and score after trainig
  verbose=False,          # Show messages (saving weights | dropout)
  file_prefix='mnist'     # File name prefix for weights save and load / None to disable saving weights
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
The NeuralNetwork can be used for any classificatoin problems. The example implementation comes with mnist dataset.

Many Thanks to Mr. Andrew Ng for his Tutorial.
