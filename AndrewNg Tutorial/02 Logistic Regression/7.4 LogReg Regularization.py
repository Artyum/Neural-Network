import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import json
import codecs

def calc_h(x, t):
    # print(x[:10])
    # print(t)
    # exit()
    a = np.array(np.dot(x, t), dtype=np.float64)
    h = 1.0 / (1.0 + np.exp(-a))
    # print(h)
    # print(h.shape)
    # exit()
    return h

def calc_cost(h, y, r):
    m = y.shape[0]
    j = y*np.log(h) + (1-y)*np.log(1-h)
    j = -j.sum()/m + r
    # print(j)
    # exit()
    return j

def logostic_regression(x, y, learn, alpha, lambda_, precision):
    # print(x)
    # exit()

    print('\nLearning', learn)
    y[y != learn] = 0.0
    y[y == learn] = 1.0

    # j_ind = np.array([])
    # j_val = np.array([])

    m = x.shape[0]
    features_num = x.shape[1]
    t = np.random.rand(features_num, 1)
    # print(m)
    # print(t)
    # exit()

    j_pre = 999
    for epoch in range(999999):
        h = calc_h(x, t)

        r = (lambda_/(2*m)) * (t[1:]**2).sum()
        j = calc_cost(h, y, r)

        if epoch % 1000 == 0:
            # print('Epoch: {0}  |  Cost: {1}'.format(epoch, j))
            print('.', end='')
            # j_ind = np.append(j_ind, epoch)
            # j_val = np.append(j_val, j)

        # Exit when the end of training
        if math.fabs(j_pre - j) < precision:
            print('\nEpoch: {0}  |  Cost: {1}'.format(epoch, j))
            break
        j_pre = j

        # print(h)
        # print(y)
        # exit()
        err = h - y
        dt = err * x
        dt = dt.sum(axis=0)  # Sum by columns
        dt = dt.reshape(-1, 1)

        t = t - alpha * (1/m * dt + (lambda_/m)*t)

        # plt.plot(j_ind, j_val)
    return t

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
rawdata = pd.read_csv('../DataSet/Iris/iris.data', names=headers)
# print(rawdata.describe())
# print(rawdata.head(5))
# exit()

# Shuffle
rawdata = shuffle(rawdata)

x_train = rawdata.iloc[:, :-1].astype(float)
y_train = rawdata.iloc[:, -1:]

# Feature scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

# Data preparation
x_train = np.array(x_train)
y_train = np.array(y_train)

# Adding the bias column = 1
x_train = np.insert(x_train, 0, 1, axis=1)    # Wstawienie kolumnty 1 "bias"

# print(x_train[:5])
# print(y_train[:5])
# print(x_train.shape)
# print(y_train.shape)
# exit()

alpha_ = 0.03
lambda_ = 0.2
precision_ = 1e-8

t_setosa = logostic_regression(x_train, y_train.copy(), learn='Iris-setosa', alpha=alpha_, lambda_=lambda_, precision=precision_)
t_versicolor = logostic_regression(x_train, y_train.copy(), learn='Iris-versicolor', alpha=alpha_, lambda_=lambda_, precision=precision_)
t_virginica = logostic_regression(x_train, y_train.copy(), learn='Iris-virginica', alpha=alpha_, lambda_=lambda_, precision=precision_)

# Verification
#    1. Age of patient at time of operation (numerical)
#    2. Patient's year of operation (year - 1900, numerical)
#    3. Number of positive axillary nodes detected (numerical)
#    4. Survival status (class attribute)
#          0 = the patient survived 5 years or longer
#          1 = the patient died within 5 year

x = np.array([               # TRUE             |  Iris-setosa    Iris-versicolor   Iris-virginica
    [5.1, 3.7, 1.5, 0.4],    # Iris-setosa      |  0.937          0.106             0.002
    [5.1, 2.5, 3.0, 1.1],    # Iris-versicolor  |  0.167          0.473             0.060
    [6.3, 3.4, 5.6, 2.4],    # Iris-virginica   |  0.003          0.244             0.890
    [4.9, 4.0, 1.1, 0.8],    # ?                |  0.937          0.062             0.003
    [5.2, 2.2, 3.2, 0.9],    # ?                |  0.139          0.600             0.051
    [6.6, 3.2, 5.3, 2.6],    # ?                |  0.002          0.287             0.937
    ])
x = scaler.transform(x)
x = np.insert(x, 0, 1, axis=1)

np.set_printoptions(precision=2, suppress=True)

print('\nIris-setosa\n', calc_h(x, t_setosa))
print('\nIris-versicolor\n', calc_h(x, t_versicolor))
print('\nIris-virginica\n', calc_h(x, t_virginica))
