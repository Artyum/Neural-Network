# Machine Learning according to Andrew Ng's Tutorial

This repository contains Python code which I've developed according to Andrew Ng Tutorial series on YouTube. The series is available on "Artificial Intelligence - All in One" channel. The first episode of the series can be found [here](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=1). The filenames correspodns to lecture's numbers.

# Neural Network

This package includes also feed-forward Neural Network implementation in Python. It was created according to Andrew Ng's Tutorial. Additionally few features were added for learning purposes.

[Lecture 9.2 — Neural Networks Learning | Backpropagation Algorithm](https://www.youtube.com/watch?v=x_Eamf8MHwU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=51)<br/>
[Lecture 9.3 — Neural Networks Learning | Backpropagation Intuition](https://www.youtube.com/watch?v=mOmkv5SI9hU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=52)

This implementation includes:
* Logistic cost function,
* Sigmoid activation function,
* Backpropagation,
* Bias gradient calculation,
* Regularization,
* Dropout,
* Split dataset to mini-batches,

You also can:
* Create any number of layers with any number of perceptrons,
* Import png files and predict self-painted samples (mnist format 28x28px),
* Save and load weights,
* Evaluate model,
* Print weights,
* Plot training progress,
* Automatically save weights at best score,
* Automatically stop training after predefined epoch number (2 parameteres: max_epoch, patience),

This Neural Network can be used for any classification problem. The example implementation comes with mnist dataset, but you can feed any other dataset you would like.
To get it working you will need [mnist-in-csv](https://www.kaggle.com/oddrationale/mnist-in-csv).

Additionally I've uploaded some custom made samples for testing. You can make your own samples in any graphical application as MsPaint.

The data structure used by functions is very close to described in A.Ng's tutorial with exceptions:

ANgT:
* The input layer is counted as first layer,
* Weights and biases of layer 'i' connects layers 'i' and 'i+1'

This implementation:
* The input layer is 'virtual' and is not considered as layer in code. It is rather a input vector of features. First layer (with index 0) is the first hidden layer,
* Weights biases of layer 'i' connects layer 'i-1' and 'i'. For the first layer (index 0) the weights are multiplied by input features.

The input vector of one sample has shape (784,1) - one column of X features. The whole training dataset shape is (784,60000) etc.
The output vector is also a 1 column array for each test sample.

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

The example evaluation of this Neural Network implementation on mnist test dataset (10'000 samples):</br>

Model: 784->800->10<br/>
Accuracy: 0.9844 (9844)<br/>
Loss: 0.0157 (156)<br/>

Model: 784->300->300->300->10<br/>
Accuracy: 0.9842 (9842)<br/>
Loss: 0.0158 (158)<br/>

Pretrained weights included for network 784->800->10:<br/>
mnist_784_800_10_l0.npy<br/>
mnist_784_800_10_l1.npy<br/>

I try to keep the nomenclature close to that used in keras framework, so I hope it shouldn't be to much confusing.

If you have any suggestions or find a bug please let me know.

Many Thanks to Mr. Andrew Ng for his Tutorial.
