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
import time


# import keyboard
# from pynput.keyboard import Key, Listener

def v2a(a):
    return a.reshape(1, -1)


class NeuralNetwork:
    """
    Input layer is virtual
    Last added layer is output layer
    """

    def __init__(self, features_num, max_epoch=1000, train_rate=0.05, accuracy=1e-5, regularization_rate=0.0, batch_size=0, dropout=0.0, drop_epoch=100):
        self.layers_num = 0
        self.features_num = features_num
        self.max_epoch = max_epoch
        self.train_rate = train_rate
        self.accuracy = accuracy
        self.batch_size = batch_size
        self.dropout = dropout
        self.drop_epoch = drop_epoch
        self.layers = np.array([])
        self.cost = 0

        if regularization_rate < 0.0:
            self.regularization_rate = 0.0
        else:
            self.regularization_rate = regularization_rate

    def add_layer(self, neurons_num, weights=None, activation='sigmoid'):
        if self.layers_num == 0:
            # First hidden layer
            layer = Layer(inputs_num=self.features_num, neurons_num=neurons_num, weights=weights)
        else:
            # Next hidden layers
            layer = Layer(inputs_num=self.layers[self.layers_num - 1].neurons_num, neurons_num=neurons_num, weights=weights)

        self.layers = np.append(self.layers, layer)
        self.layers_num += 1

    def feed_forward(self, x):
        for i in range(self.layers_num):
            if i == 0:
                self.layers[i].calc_activation_values(x)  # Pass input values
            else:
                self.layers[i].calc_activation_values(self.layers[i - 1].a)  # Pass values from previous layer
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

    def train(self, x_train, y_train, x_test, y_test):
        _time_total = time.time()

        plot = 0

        if plot:
            px = np.array([])
            pcost = np.array([])
            pscore = np.array([])

        pre_cost = 9999.0
        pre_score = -1.0
        m_total = x_train.shape[1]  # Total number of samples
        end = 0
        epoch = 0

        if self.batch_size > m_total:
            self.batch_size = m_total

        # for epoch in range(self.max_epoch + 1):
        while end < self.max_epoch:

            # Setup batch indexes
            batch_from = 0
            batch_to = self.batch_size

            # Print out progress
            if epoch % 1 == 0:
                self.feed_forward(x_train)
                self.calc_cost(y_train)

                score = self.test(x_test, y_test)

                # Set initial pre_score
                if pre_score == -1.0:
                    pre_score = score

                if score > pre_score:
                    self.save_weights()
                    pre_score = score
                    end = 0

                    if plot:
                        px = np.append(px, epoch)
                        pcost = np.append(pcost, self.cost)
                        pscore = np.append(pscore, score)
                else:
                    end += 1

                print('Epoch: {0}\t|\tCost: {1}\t|\tScore: {2}\t|\tMax score: {4}\t|\tEnd: {3}'.format(epoch, self.cost, score, end, pre_score))

                # Check if training is finished
                if math.fabs(pre_cost - self.cost) < self.accuracy:
                    print('Epoch: {0}  |  Cost: {1}'.format(epoch, self.cost))
                    break
                pre_cost = self.cost

            # Dropout
            if self.dropout > 0.0 and (self.max_epoch - end) % self.drop_epoch == 0 and end > 0:
                print('Dropout', self.dropout)
                for i in range(self.layers_num):
                    for j in range(self.layers[i].weights.shape[0]):
                        for k in range(self.layers[i].weights.shape[1]):
                            if np.random.random() <= self.dropout:
                                self.layers[i].weights[j, k] = 0.0
                                # epsilon = 0.7
                                # self.layers[i].weights[j, k] = np.random.random() * 2.0 * epsilon - epsilon

            # Batch loop
            while batch_from < m_total:
                # Split data to batch
                xbatch = x_train[:, batch_from:batch_to]
                ybatch = y_train[:, batch_from:batch_to]

                # Number of samples in batch
                m = xbatch.shape[1]

                self.feed_forward(xbatch)

                # Clear deltasum for all layers
                for i in range(self.layers_num):
                    self.layers[i].deltasum = None

                # Backpropagation
                for i in reversed(range(self.layers_num)):
                    # print('Layer', i)
                    if i == self.layers_num - 1:
                        self.layers[i].delta = self.output - ybatch
                    else:
                        w = self.layers[i + 1].weights[:, 1:]  # Cut bias weights
                        d = self.layers[i + 1].delta
                        c = np.dot(w.T, d)
                        a = self.layers[i].a
                        s = a * (1 - a)
                        self.layers[i].delta = c * s

                    if i == 0:
                        deltasum = np.dot(self.layers[i].delta, xbatch.T)
                    else:
                        deltasum = np.dot(self.layers[i].delta, self.layers[i - 1].a.T)

                    if self.layers[i].deltasum is None:
                        self.layers[i].deltasum = deltasum
                    else:
                        self.layers[i].deltasum += deltasum

                    # print('delta\n', self.layers[i].delta)
                    # print('deltasum\n', self.layers[i].deltasum)

                # Weights update
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
                        r[:, 0] = 0.0  # Exclude regularization for bias weights / 0 first column
                    else:
                        r = 0.0

                    # Update all biases and weights in layer
                    self.layers[i].weights -= (self.train_rate * (d_weights / m) + r)
                    # print('weights\n', self.layers[i].weights)

                batch_from = batch_to
                batch_to += self.batch_size
                if batch_to > m_total:
                    batch_to = m_total

            epoch += 1

        print('Total time [sec]:', time.time() - _time_total)

        if plot:
            plt.plot(px, pcost)
            plt.plot(px, pscore)
            plt.show()

    def test(self, x, y):
        network.feed_forward(x)
        t = arr2digit(y)
        p = arr2digit(network.finalize)
        return calc_score(t, p)

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
            print('Weights Layer:{0} | In:{1} | Neurons:{2}'.format(i, self.layers[i].inputs_num, self.layers[i].neurons_num))
            print(self.layers[i].weights)

    def fname(self, l):
        n = '9.3_v3_' + str(self.features_num)
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
            epsilon = 0.7
            self.weights = np.random.rand(self.neurons_num, self.inputs_num + 1) * 2.0 * epsilon - epsilon  # inputs_num+1 for bias
        else:
            self.weights = weights

    def calc_activation_values(self, x):
        # Inserting bias row = 1
        x = np.insert(x, 0, 1, axis=0)
        z = np.dot(self.weights, x)

        # Sigmoid (logistic) activation function
        self.a = 1.0 / (1.0 + np.exp(-z))


def prepare_dataset(file):
    rawdata = pd.read_csv(file, header=None)
    # print(rawdata.describe())
    # print(rawdata.head(5))
    # exit()

    # Shuffle
    rawdata = shuffle(rawdata)

    # Separate last column of dataset -> features from class
    x = np.array(rawdata.iloc[:, :-1].astype(float))

    y = []
    for i in rawdata.iloc[:, 16]:
        if i == 0: a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 1: a = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 2: a = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if i == 3: a = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if i == 4: a = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if i == 5: a = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if i == 6: a = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if i == 7: a = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if i == 8: a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if i == 9: a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y.append(a)
    y = np.asarray(y)
    # print(y)
    # exit()

    return x, y


def arr2digit(a):
    arr = []
    for i in a.T:
        if i.sum() != 1:
            d = 0
        else:
            d = 1 * i[0] + 2 * i[1] + 3 * i[2] + 4 * i[3] + 5 * i[4] + 6 * i[5] + 7 * i[6] + 8 * i[7] + 9 * i[8] + 10 * i[9]
        arr.append(d - 1)
    return np.array(arr)


def calc_score(true, predict):
    m = len(true)
    a = 0
    for i in range(m):
        if true[i] == predict[i]:
            a += 1
    return a / m


def print_score(true, predict):
    m = len(true)
    a = 0
    for i in range(m):
        if true[i] == predict[i]:
            a += 1
    print('\nTest samples:', m)
    print('Correct:', a)
    print('False:', m - a)
    print('Ratio:', a / m)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

np.set_printoptions(precision=2, suppress=True)
# np.set_printoptions(suppress=True)

x_train, y_train = prepare_dataset('../../DataSet/pendigits/pendigits.train')
x_test, y_test = prepare_dataset('../../DataSet/pendigits/pendigits.test')
x_all, y_all = prepare_dataset('../../DataSet/pendigits/pendigits.all')

# Feature scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_all = scaler.transform(x_all)

network = NeuralNetwork(features_num=16, max_epoch=2000, train_rate=1, accuracy=1e-10, regularization_rate=0, batch_size=10, dropout=0.25, drop_epoch=100)
network.add_layer(neurons_num=150)
network.add_layer(neurons_num=10)
network.add_layer(neurons_num=150)
network.add_layer(neurons_num=10)

x_train = x_train.T
y_train = y_train.T
x_all = x_all.T

total_samples = x_test.shape[0]
test_samples = total_samples
x_test = x_test[:test_samples].T
y_test = y_test[:test_samples].T
y_all = y_all.T

# print('x_train\n', x_train)
# print('y_train\n', y_train)
# exit

# TRAIN
# network.train(x_train, y_train, x_test, y_test)

# TEST
network.load_weights()

# print('Total test samples:', total_samples)
# print('Test samples:', test_samples)

network.feed_forward(x_test)
true = arr2digit(y_test)
predict = arr2digit(network.finalize)
print_score(true, predict)

network.feed_forward(x_all)
true = arr2digit(y_all)
predict = arr2digit(network.finalize)
print_score(true, predict)

# print('True output\n', true)
# print('Network finalize\n', predict)

"""
In |  L1 |  L2 |  L3 | Out                         |  Err ratio / all  / true / false
-------------------------------------------------------------------------------------
16 |  30 |  30 |  10          2, 1e-10, None, 100  |  0.9674099 / 3498 / 3384 / 114
16 |  30 |  30 |  20 | 10     2, 1e-10, None, 100  |  0.9685534 / 3498 / 3388 / 110
16 |  40 |  40 |  40 | 10     5, 1e-10, None, 100  |  0.9714122 / 3498 / 3398 / 100
16 |  24 |  24 |  10          1, 1e-10, None, 100  |  0.9728416 / 3498 / 3403 / 95
16 |  60 |  60 |  10          2, 1e-10, None, 100  |  0.9731275 / 3498 / 3404 / 94
16 | 120 | 120 |  10          2, 1e-10, None, 100  |  0.9739851 / 3498 / 3407 / 91
16 | 150 | 150 | 150 | 10     2, 1e-10, None, 100  |  0.9745568 / 3498 / 3409 / 89
16 |  80 |  80 |  10          2, 1e-10, None, 100  |  0.9748427 / 3498 / 3410 / 88
16 | 100 | 100 |  10          2, 1e-10, None, 10   |  0.9779874 / 3498 / 3421 / 77
16 | 120 | 120 |  10          2, 1e-10, None, 10   |  0.9791309 / 3498 / 3425 / 73-75
16 | 160 | 160 |  10        1.5, 1e-10, None, 10   |  0.9799885 / 3498 / 3428 / 70
16 | 150 |  10 | 150 |  10    1, 1e-10, None, 10   |  0.9817038 / 3498 / 3434 / 64-66

"""

# print('Network output\n', network.output)
# network.print_weights()
