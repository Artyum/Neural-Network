# Lecture 9.2 — Neural Networks Learning | Backpropagation Algorithm — [ Machine Learning | Andrew Ng]
# https://www.youtube.com/watch?v=x_Eamf8MHwU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=51

# Lecture 9.3 — Neural Networks Learning | Backpropagation Intuition
# https://www.youtube.com/watch?v=mOmkv5SI9hU&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=52

# 09: Neural Networks - Learning
# http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html

# Principles of training multi-layer neural network using backpropagation
# http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

# Matrix Multiplication
# http://matrixmultiplication.xyz/

import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from PIL import Image
import os
import glob


class NeuralNetwork:
    def __init__(self, input_dim, max_epoch=1000, train_rate=0.05, accuracy=1e-5, regularization=0.0, batch_size=0, patience=100, dropout=0.0, drop_epoch=100, plot=False, verbose=False, file_prefix=None):
        self.layers_num = 0
        self.input_dim = input_dim
        self.max_epoch = max_epoch
        self.train_rate = train_rate
        self.accuracy = accuracy
        self.regularization = regularization
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.drop_epoch = drop_epoch
        self.plot = plot
        self.layers = np.array([])
        self.verbose = verbose
        self.file_prefix = file_prefix
        self.cost = 0

        if self.max_epoch < 0: self.max_epoch = 0
        if self.accuracy <= 0.0: self.accuracy = 1e-10
        if self.regularization < 0.0: self.regularization = 0.0
        if self.file_prefix == '': self.file_prefix = None

    def add_layer(self, units, epsilon=0.707, weights=None):
        """
        Input layer is virtual
        Last added layer is output layer
        """
        if self.layers_num == 0:
            # First hidden layer
            layer = Layer(inputs_num=self.input_dim, neurons_num=units, epsilon=epsilon, weights=weights)
        else:
            # Another hidden layers
            layer = Layer(inputs_num=self.layers[self.layers_num - 1].neurons_num, neurons_num=units, epsilon=epsilon, weights=weights)

        self.layers = np.append(self.layers, layer)
        self.layers_num += 1

    def feed_forward(self, x):
        for i in range(self.layers_num):
            if i == 0:
                # For first layer pass input values
                self.layers[i].calc_activation_values(x)
            else:
                # Pass values from previous layer
                self.layers[i].calc_activation_values(self.layers[i - 1].a)

    def predict(self, x):
        self.feed_forward(x)
        return self.output

    def calc_cost(self, y):
        m = y.shape[1]

        j = y * np.log(self.output) + (1 - y) * np.log(1 - self.output)
        j = -j.sum() / m

        # Generalization | Sum of all thetas excluding bias
        if self.regularization > 0.0:
            t_sum = 0
            for i in range(self.layers_num):
                s = (self.layers[i].weights ** 2).sum(axis=0)  # Sum over columns
                s = s.sum() - s[0]  # Sum all weights / exclude bias
                t_sum += s
            j += self.regularization / (2.0 * m) * t_sum

        self.cost = j

    def fit(self, x_train, y_train, x_test, y_test):
        if self.plot:
            px = []
            pcost = []
            pscore = []

        pre_cost = 9999.0
        pre_score = -1.0
        m_total = x_train.shape[1]  # Total number of samples
        end = 0
        epoch = 0

        if self.batch_size > m_total:
            self.batch_size = m_total

        # for epoch in range(self.max_epoch + 1):
        while end < self.patience and epoch < self.max_epoch:
            time_start = time.time()

            # Setup batch indexes
            batch_from = 0
            batch_to = self.batch_size

            # Dropout
            # If dropout 0, on/off switch
            # If epoch > 0, because at 0 epoch after weight load we want the score exactly as it was saved
            # (self.max_epoch - end) % self.drop_epoch == 0 and end > 0 to drop every drop_epoch
            if self.dropout > 0.0 and epoch > 0 and (self.max_epoch - end) % self.drop_epoch == 0:
                if self.verbose: print('Dropout', self.dropout)
                for i in range(self.layers_num):
                    for j in range(self.layers[i].weights.shape[0]):
                        for k in range(self.layers[i].weights.shape[1]):
                            if np.random.random() <= self.dropout:
                                self.layers[i].weights[j, k] = 0.0

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
                    if i == self.layers_num - 1:
                        self.layers[i].delta = self.output - ybatch
                    else:
                        w = self.layers[i + 1].weights[:, 1:]  # Cut bias weights
                        d = self.layers[i + 1].delta
                        c = np.dot(w.T, d)
                        a = self.layers[i].a

                        # Sigmoid derivative
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

                # Weights and biases update
                for i in range(self.layers_num):
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

                    # Regularization
                    if self.regularization > 0.0:
                        r = self.regularization * self.layers[i].weights
                        r[:, 0] = 0.0  # Exclude regularization for bias weights / 0 first column
                    else:
                        r = 0.0

                    # Update all biases and weights in layer
                    self.layers[i].weights -= (self.train_rate * (d_weights / m) + r)

                batch_from = batch_to
                batch_to += self.batch_size

            # Calculate cost function
            self.feed_forward(x_train)
            self.calc_cost(y_train)

            # Evaluate and print progress
            score = self.evaluate(x_test, y_test, verbose=False)

            if pre_score == -1.0:
                pre_score = score

            if epoch > 0 and score > pre_score:
                self.save_weights()
                pre_score = score
                end = 0

                if self.plot:
                    px.append(epoch)
                    pcost.append(self.cost)
                    pscore.append(score)
            else:
                end += 1

            # Epoch time
            epoch_time = time.time() - time_start

            print('Epoch: ' + str(epoch) + '/' + str(self.max_epoch) + '\t|\tCost: ' + str(round(self.cost, 5)) + '\t|\tAccuracy: ' + str(round(score, 5)) +
                  '\t|\tMax accuracy: ' + str(round(pre_score, 5)) + ' (loss: ' + str(round((1.0 - pre_score) * 100.0, 5)) + '%)\t|\tPatience: ' + str(self.patience - end) +
                  '\t|\tEpoch time: ' + str(round(epoch_time, 2)))

            # Check if training is finished
            if math.fabs(pre_cost - self.cost) < self.accuracy:
                print('Epoch: {0}  |  Cost: {1}'.format(epoch, self.cost))
                break
            pre_cost = self.cost

            epoch += 1

        if self.plot:
            plt.plot(px, pcost)
            plt.plot(px, pscore)
            plt.show()

    def evaluate(self, x, y, verbose=True):
        prediction = arr2digit(network.predict(x))
        true_val = arr2digit(y)

        # Predicted digits divided by number of all samples
        m = len(true_val)
        cnt = 0
        for i in range(m):
            if true_val[i] == prediction[i]:
                cnt += 1
        accuracy = cnt / m
        loss = 1.0 - accuracy
        if verbose:
            print('Test samples:', m)
            print('Accuracy: ' + str(round(accuracy, 5)) + ' (' + str(cnt) + ')')
            print('Loss: ' + str(round(loss, 5)) + ' (' + str(m - cnt) + ')')
        return accuracy

    def save_weights(self):
        if self.file_prefix is not None:
            if self.verbose: print('Saving weights')
            for i in range(self.layers_num):
                np.save(self.fname(i), self.layers[i].weights)

    def load_weights(self):
        if self.file_prefix is not None:
            if self.verbose: print('Loading weights')
            for i in range(self.layers_num):
                self.layers[i].weights = np.load(self.fname(i))

    def fname(self, l):
        n = self.file_prefix + '_' + str(self.input_dim)
        for i in range(self.layers_num):
            n += '_' + str(self.layers[i].neurons_num)
        n = n + '_l' + str(l) + '.npy'
        return n

    def print_weights(self, layer=None):
        for i in range(self.layers_num):
            if i == layer or layer is None:
                print('Weights Layer: {0} | Inputs: {1} | Units: {2}'.format(i, self.layers[i].inputs_num, self.layers[i].neurons_num))
                print(self.layers[i].weights)

    # Returns values of last layer
    @property
    def output(self):
        return self.layers[self.layers_num - 1].a


class Layer:
    def __init__(self, inputs_num, neurons_num, epsilon, weights=None):

        self.inputs_num = inputs_num
        self.neurons_num = neurons_num

        self.a = None
        self.delta = None
        self.deltasum = None

        if weights is None:
            # Randomize initial weights in range (-epsilon; +epsilon)
            self.weights = np.random.rand(self.neurons_num, self.inputs_num + 1) * 2.0 * epsilon - epsilon  # self.inputs_num+1: '+1' for bias
        else:
            self.weights = weights

    def calc_activation_values(self, x):
        # Inserting bias row = 1 for matrix calculations. Weights matrix contains bias values.
        x = np.insert(x, 0, 1, axis=0)
        z = np.dot(self.weights, x)

        # Activation function
        self.a = 1.0 / (1.0 + np.exp(-z))  # Sigmoid (logistic)


# Prepare one-hot array
def prepare_mnist_dataset(file, randomize=True):
    rawdata = pd.read_csv(file, header=None)

    # Shuffle
    if randomize:
        rawdata = shuffle(rawdata)

    # Separate first column of dataset -> class from features
    x = np.array(rawdata.iloc[:, 1:].astype(float))

    y = []
    for i in rawdata.iloc[:, 0]:
        if i == 0: arr = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 1: arr = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if i == 2: arr = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if i == 3: arr = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if i == 4: arr = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if i == 5: arr = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if i == 6: arr = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if i == 7: arr = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if i == 8: arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if i == 9: arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y.append(arr)
    y = np.asarray(y)

    return x, y


# Decode array to single digit
def arr2digit(a):
    ret = []
    for i in a.T:
        # Find highest predicted value for each sample
        ret.append(np.argmax(i))
    return ret


# Load all png files from specified folder, encode to list of values and save as csv. Images must be 28x28px.
# First letter of filename contains digit example: 1.png | 2test.png | 0mnist.png
def import_custom_png_images(folder, output):
    ret = ''
    for path in glob.glob(folder + '/*png'):
        s = ''
        img = Image.open(path)
        img = list(img.convert('L').getdata())
        file = os.path.basename(path)

        for x in img:
            s += ',' + str(255 - x)
        s = file[0] + s + '\n'
        ret += s

    text_file = open(output, 'w')
    text_file.write(ret)
    text_file.close()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

np.set_printoptions(precision=3, suppress=True)
# np.set_printoptions(suppress=True)

# MNIST dataset
x_train, y_train = prepare_mnist_dataset(file='../../DataSet/mnist/mnist_train.csv', randomize=True)
x_test, y_test = prepare_mnist_dataset(file='../../DataSet/mnist/mnist_test.csv', randomize=True)

# Combined mnist_train.csv and and mnist_test.csv
x_all, y_all = prepare_mnist_dataset(file='../../DataSet/mnist/mnist_all.csv', randomize=False)

# Custom test set of digits created in Paint (28x28px png)
import_custom_png_images(folder='../../DataSet/mnist/Custom', output='../../DataSet/mnist/mnist_custom.csv')
x_cust, y_cust = prepare_mnist_dataset(file='../../DataSet/mnist/mnist_custom.csv', randomize=False)

# Feature scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_all = scaler.transform(x_all)
x_cust = scaler.transform(x_cust)

# NeuralNetwork class accepts dataset shape: rows=features, cols=samples (as described in Andrew Ng's tutorial)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
x_all = x_all.T
y_all = y_all.T
x_cust = x_cust.T
y_cust = y_cust.T

# Neural Network Model
network = NeuralNetwork(input_dim=784,  # Input layer size
                        max_epoch=1000,  # Stop training after total number of epochs
                        train_rate=2.0,  # Learning rate
                        accuracy=1e-6,  # Stop training if cost function change is less than accuracy
                        regularization=0.0001,  # Regularization rate / 0 - disable regularization
                        batch_size=256,  # Mini batch size
                        patience=30,  # Stop after number of epochs countdown since best score
                        dropout=0.1,  # Dropps % of weights in every layer / 0 - disable dropout
                        drop_epoch=30,  # Dropout every 'drop_epoch' number since best score
                        plot=False,  # Enable plot of cost and score after trainig
                        verbose=True,  # Show messages (saving weights | load weights | dropout)
                        file_prefix='mnist'  # File name prefix for weights save and load / None to disable saving weights
                        )

# Hidden layers
network.add_layer(units=800, epsilon=0.7)

# Last added layer is output layer
network.add_layer(units=10, epsilon=0.7)

# # # # # # # #
# TRAIN

network.load_weights()  # Uncomment to load weights to continue training
network.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# # # # # # # #
# TEST

network.load_weights()

# # # # # # # #
# PRINT RESULTS

print('\n* Evaluate x_train')
network.evaluate(x_train, y_train)

print('\n* Evaluate x_test')
network.evaluate(x_test, y_test)

print('\n* Evaluate x_all')
network.evaluate(x_all, y_all)

print('\n* Evaluate x_cust')
network.evaluate(x_cust, y_cust)
print('True values', arr2digit(y_cust))
print('Prediction ', arr2digit(network.predict(x_cust)))
