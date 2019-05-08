# 8.4 https://www.youtube.com/watch?v=iPNN805konI&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=46

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def v2a(a):
    return a.reshape(1, -1)

class NeuralNetwork:
    """
    Input layer is virtual
    Last added layer is output layer
    """
    def __init__(self, features_num, learning_rate=0.05):
        self.layers_num = 0
        self.features_num = features_num
        self.learning_rate = learning_rate
        self.layers = np.array([])
        self.cost = 0

    def add_layer(self, neurons_num, weights=None):
        if self.layers_num == 0:
            # First hidden layer
            layer = Layer(inputs_num=self.features_num, neurons_num=neurons_num, weights=weights)
        else:
            # Next hidden layers
            layer = Layer(inputs_num=self.layers[self.layers_num-1].neurons_num, neurons_num=neurons_num, weights=weights)

        self.layers = np.append(self.layers, layer)
        self.layers_num += 1

    def feed_forward(self, x):
        for i in range(self.layers_num):
            if i == 0:
                self.layers[i].calc_activation_values(x)        # Pass input values
            else:
                self.layers[i].calc_activation_values(self.layers[i-1].a)       # Pass values from previous layer
            # print('Layer', i)
            # print('a\n', self.layers[i].a)

    @property
    def output(self):
        # Returns values from last (output) layer
        return self.layers[self.layers_num-1].a

class Layer:
    def __init__(self, inputs_num, neurons_num, weights=None):
        self.inputs_num = inputs_num
        self.neurons_num = neurons_num
        self.z = None
        self.a = None

        if weights is None:
            # Randomize initial weights
            self.weights = np.random.rand(self.neurons_num, inputs_num+1)       # +1 for bias
        else:
            self.weights = weights

    def calc_activation_values(self, x):
        x = np.insert(x, 0, 1, axis=0)      # Inserting bias row = 1
        # print(x)
        # print(self.weights)
        # exit()
        self.z = np.dot(self.weights, x)
        # print('z\n', self.z)
        self.a = 1.0 / (1.0 + np.exp(-self.z))      # Sigmoid (logistic) activation function
        # self.a = self.z

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

learning_set = 5

headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
rawdata = pd.read_csv('../DataSet/Iris/iris.data', names=headers)
# print(rawdata.describe())
# print(rawdata.head(5))
# exit()

# Shuffle
rawdata = shuffle(rawdata)

# rawdata['class'] = rawdata['class'].map({'Iris-setosa': np.array([[1, 0, 0]]), 'Iris-versicolor': np.array([[0, 1, 0]]), 'Iris-virginica': np.array([[0, 0, 1]])})

# Divide dataset to train and test columns
x = np.array(rawdata.iloc[:, :-1].astype(float))
y = []
for i in rawdata['class']:
    if i == 'Iris-setosa':
        a = np.array([1, 0, 0])
    elif i == 'Iris-versicolor':
        a = np.array([0, 1, 0])
    elif i == 'Iris-virginica':
        a = np.array([0, 0, 1])
    y.append(a)
y = np.asarray(y)

# Feature scaling
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Adding bias column = 1
# x = np.insert(x, 0, 1, axis=1)     # Inserting bias column = 1

# Divide dataset to training and test samples
x_train = x[:learning_set]
y_train = y[:learning_set]
x_test = x[learning_set:]
y_test = y[learning_set:]

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# print(x_train.shape)    # (120, 5)
# print(y_train.shape)    # (120, 3)
# print(x_test.shape)     # (30, 5)
# print(y_test.shape)     # (30, 3)

# Test 4 -> 3 -> 2
# w1 = np.array([[.1, .2, .3, .4], [.4, .3, .2, .1], [.2, .3, .4, .5]])
# w2 = np.array([[.3, .2, .1], [.1, .2, .3]])
# network = NeuralNetwork(features_num=4)
# network.add_layer(3, weights=w1)
# network.add_layer(2, weights=w2)
# # x = v2a(np.array([1, 2, 3, 4])).T
# x = np.array([[.1, .2, .3, .4], [.1, .2, .3, .4]]).T
# print('x\n', x)
# network.feed_forward(x)
# print('h\n', network.output)
# exit()

network = NeuralNetwork(features_num=4, learning_rate=0.05)
network.add_layer(3)
network.add_layer(2)

x = x_train.T
y = y_train.T
print('x\n', x)
print('y\n', y)
network.feed_forward(x)
print('h\n', network.output)
