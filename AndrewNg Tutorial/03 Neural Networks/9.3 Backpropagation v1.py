# 9.3 https://www.youtube.com/watch?v=mOmkv5SI9hU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=52

# 09: Neural Networks - Learning
# http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html

# Principles of training multi-layer neural network using backpropagation
# http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
# http://galaxy.agh.edu.pl/~vlsi/AI/backp_t/backprop.html

# Matrix Multiplication
# http://matrixmultiplication.xyz/

import numpy as np
import pandas as pd
import math
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

    def __init__(self, features_num, training_rate=0.05, accuracy=1e-5, regularization_rate=0.0):
        self.layers_num = 0
        self.features_num = features_num
        self.training_rate = training_rate
        self.accuracy = accuracy
        self.regularization_rate = regularization_rate
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
                self.layers[i].calc_activation_values(x)  # Pass input values
            else:
                self.layers[i].calc_activation_values(self.layers[i-1].a)  # Pass values from previous layer
            # print('Layer', i)
            # print('a\n', self.layers[i].a)

    def calc_cost(self, y):
        m = y.shape[1]

        j = y * np.log(self.output) + (1 - y) * np.log(1 - self.output)

        # print('j\n', j)
        # print('m\n', m)

        j = -j.sum() / m

        # Generalization | Sum of all thetas excluding bias
        if self.regularization_rate > 0.0:
            t_sum = 0
            for i in range(self.layers_num):
                # print(i)
                # print(self.layers[i].weights)
                s = (self.layers[i].weights ** 2).sum(axis=0)  # Sum over columns
                s = s.sum() - s[0]  # Sum all weights beside bias
                # print('s\n', s)
                t_sum += s
            # print('t_sum\n', t_sum)
            j += self.regularization_rate / (2.0 * m) * t_sum

        self.cost = j

    def train(self, x, y):
        # j_ind = np.array([])
        # j_val = np.array([])

        pre_cost = 0
        for epoch in range(999999999):
            self.feed_forward(x)
            self.calc_cost(y)

            if math.fabs(pre_cost-self.cost) < self.accuracy:
                print('Epoch: {0}  |  Cost: {1}'.format(epoch, self.cost))
                break;
            pre_cost = self.cost

            if epoch % 1000 == 0:
                print('Epoch: {0}  |  Cost: {1}'.format(epoch, self.cost))
                # j_ind = np.append(j_ind, epoch)
                # j_val = np.append(j_val, self.cost)

            for i in range(self.layers_num):
                self.layers[i].deltasum = None

            for i in reversed(range(self.layers_num)):
                # print('Layer', i)
                if i == self.layers_num-1:
                    self.layers[i].delta = self.output - y
                else:
                    w = self.layers[i+1].weights[:, 1:]  # Cut bias weights
                    d = self.layers[i+1].delta
                    # print('w\n', w)
                    # print('w.T\n', w.T)
                    # print('d\n', d)
                    c = np.dot(w.T, d)
                    # print('c\n', c)
                    a = self.layers[i].a
                    s = a * (1 - a)
                    self.layers[i].delta = c * s

                if i == 0:
                    deltasum = np.dot(self.layers[i].delta, x.T)
                else:
                    deltasum = np.dot(self.layers[i].delta, self.layers[i-1].a.T)

                if self.layers[i].deltasum is None:
                    self.layers[i].deltasum = deltasum
                else:
                    self.layers[i].deltasum += deltasum

                # print('delta\n', self.layers[i].delta)
                # print('deltasum\n', self.layers[i].deltasum)

            # Weights update
            m = x.shape[1]   # Number of samples
            for i in range(self.layers_num):
                # print('Layer', i)

                # For bias using layers[i].delta without multiplied by input in given layer
                # Bias matrix (M x N)
                # N kolumns - number of samples
                # M rows - values for bias weights. Number of rows = number of neurons in next layer
                d_bias = self.layers[i].delta.sum(axis=1).reshape(-1, 1)

                # Weights matrix (M x N) without bias
                # M rows - neurons number in next layer
                # N columns - neurons number in current layer
                d_weights = self.layers[i].deltasum

                # Combine bias and weights partial derivatives into one matrix
                d_weights = np.concatenate((d_bias, d_weights), axis=1)

                # print('d_bias\n', d_bias)
                # print('d_weights\n', d_weights)
                # print('concat\n', concat)

                # Regularization
                if self.regularization_rate > 0.0:
                    r = self.regularization_rate * self.layers[i].weights
                    r[:, 0] = 0.0    # Exclude regularization for bias weights / 0 first column
                else:
                    r = 0.0

                # Update all biases and weights in layer
                self.layers[i].weights -= (self.training_rate * (d_weights/m) + r)
                # print('weights\n', self.layers[i].weights)

        # plt.plot(j_ind, j_val)
        # plt.show()

    def save_weights(self):
        for i in range(self.layers_num):
            np.save(self.fname(i), self.layers[i].weights)
            # print(self.layers[i].weights)

    def load_weights(self):
        for i in range(self.layers_num):
            self.layers[i].weights = np.load(self.fname(i))
            # print(self.layers[i].weights)

    def print_weights(self):
        for i in range(self.layers_num):
            print('Weights layer:', i)
            print(self.layers[i].weights)

    def fname(self, l):
        n = '9.3_' + str(self.features_num)
        for i in range(self.layers_num):
            n += '_' + str(self.layers[i].neurons_num)
        n = n + '_layer' + str(l) + '_n' + str(self.layers[l].neurons_num) + '.npy'
        return n

    @property
    def output(self):
        # Returns values from last (output) layer
        return self.layers[self.layers_num - 1].a

    @property
    def finalize(self):
        o = np.copy(self.output)
        o[o >= 0.5] = 1
        o[o < 0.5] = 0
        return o.astype(int)

class Layer:
    def __init__(self, inputs_num, neurons_num, weights=None):

        self.inputs_num = inputs_num
        self.neurons_num = neurons_num
        # print(self.inputs_num)
        # print(self.neurons_num)
        # exit()

        self.a = None
        self.delta = None
        self.deltasum = None

        if weights is None:
            # Randomize initial weights in range (-epsilon; +epsilon)
            epsilon = 1.5
            self.weights = np.random.rand(self.neurons_num, self.inputs_num+1) * 2.0 * epsilon - epsilon  # inputs_num+1 for bias
        else:
            self.weights = weights

    def calc_activation_values(self, x):
        x = np.insert(x, 0, 1, axis=0)  # Inserting bias row = 1
        # print('x\n', x)
        # print('weights\n', self.weights)
        # exit()
        z = np.dot(self.weights, x)
        # print('z\n', z)
        self.a = 1.0 / (1.0 + np.exp(-z))  # Sigmoid (logistic) activation function
        # self.a = z
        # exit()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


np.set_printoptions(precision=2, suppress=True)
# np.set_printoptions(suppress=True)

learning_set = 120

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
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)

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
# x = np.array([[.1, .2, .3, .4], [.1, .2, .3, .4]]).T
# y = np.array([[2, 2], [2, 2]]).T
# w1 = np.array([[.9, .1, .2, .3, .4], [.9, .4, .3, .2, .1], [.9, .2, .3, .4, .5]])
# w2 = np.array([[.9, .3, .2, .1], [.9, .1, .2, .3]])
# network = NeuralNetwork(features_num=4)
# network.add_layer(neurons_num=3, weights=w1)
# network.add_layer(neurons_num=2, weights=w2)
# # for i in range(network.layers_num):
# #     print(i)
# #     print('weights.shape', network.layers[i].weights.shape)
# # network.feed_forward(x)
# # print('h\n', network.output)
# network.train(x, y)
# exit()

network = NeuralNetwork(features_num=4, training_rate=0.1, accuracy=1e-7, regularization_rate=5e-4)
network.add_layer(neurons_num=12)
network.add_layer(neurons_num=12)
network.add_layer(neurons_num=3)

# for i in range(network.layers_num):
#     print('Layer:', i)
#     print(network.layers[i].weights.shape)
# exit()

x = x_train.T
y = y_train.T
# print('x\n', x)
# print('y\n', y)
# network.feed_forward(x)
# print('h\n', network.output)
# network.calc_cost(y)
# print('cost\n', network.cost)

network.train(x, y)
# network.save_weights()
# network.load_weights()

# TEST

# print('Test samples:', x_test.shape[0])
test_num = 10
x = x_test[:test_num].T
y = y_test[:test_num].T
print('Test class\n', x)
print('True output\n', y)

network.feed_forward(x)
print('Network finalize\n', network.finalize)
print('Difference\n', (y-network.finalize).sum())
print('Network output\n', network.output)

# network.print_weights()

# High Variance
#   - Get more training examples
#   - Try smaller set of features (chunks)
#   - Try increasing lambda
#
# High bias
#   - Try getting additional features
#   - Try adding polynomial features
#   - Try decreasing lambda
