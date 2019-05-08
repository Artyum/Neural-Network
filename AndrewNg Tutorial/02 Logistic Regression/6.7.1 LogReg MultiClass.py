import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

def calc_h(x, t):
    # print(x[:10])
    # print(t)
    # exit()
    a = np.array(np.dot(x, t), dtype=np.float32)
    h = 1.0 / (1.0 + np.exp(-a))
    # print(h)
    # print(h.shape)
    # exit()
    return h

def calc_cost(h, y):
    m = y.shape[0]
    j = y*np.log(h) + (1-y)*np.log(1-h)
    j = -j.sum()/m
    # print(j)
    # exit()
    return j

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
features_num = x_train.shape[1]

# Adding the bias column = 1
x_train = np.insert(x_train, 0, 1, axis=1)

# print(x_train[:5])
# print(y_train[:5])
# print(x_train.shape)
# print(y_train.shape)
# exit()

# # # # # # # # # # #
# Switching classes

# Iris-setosa
# Iris-versicolor
# Iris-virginica

y_train[y_train != 'Iris-setosa'] = 0.0
y_train[y_train == 'Iris-setosa'] = 1.0
# y_train[y_train != 'Iris-versicolor'] = 0.0
# y_train[y_train == 'Iris-versicolor'] = 1.0
# y_train[y_train != 'Iris-virginica'] = 0.0
# y_train[y_train == 'Iris-virginica'] = 1.0

# print(y_train)
# exit()

t = np.random.rand(features_num+1, 1)*2-1   # +1 for bias
# print(t)
# exit()

x = x_train
y = y_train

m = x.shape[0]
alpha = 0.5

j_ind = np.array([])
j_val = np.array([])

j_pre = 999
for epoch in range(999999):
    h = calc_h(x, t)

    j = calc_cost(h, y)
    print('Epoch: {0}  |  Cost: {1}'.format(epoch, j))

    j_ind = np.append(j_ind, epoch)
    j_val = np.append(j_val, j)

    # Exit when the end of training
    if math.fabs(j_pre - j) < 0.0001:
        print('Epoch: {0}  |  Cost: {1}'.format(epoch, j))
        break
    j_pre = j

    # print(h)
    # print(y)
    # exit()
    err = h - y
    dt = err * x
    dt = dt.sum(axis=0)  # Sum by columns
    dt = dt.reshape(-1, 1)

    t = t - alpha / m * dt

# print('Theta\n', t)

# Verification
#    1. Age of patient at time of operation (numerical)
#    2. Patient's year of operation (year - 1900, numerical)
#    3. Number of positive axillary nodes detected (numerical)
#    4. Survival status (class attribute)
#          0 = the patient survived 5 years or longer
#          1 = the patient died within 5 year
x = np.array([               # TRUE              | Iris-setosa     Iris-versicolor   Iris-virginica
    [5.1, 3.7, 1.5, 0.4],    # Iris-setosa       | 0.9366577       0.10634837        0.00119422
    [5.1, 2.5, 3.0, 1.1],    # Iris-versicolor   | 0.19216032      0.4851717         0.05380678
    [6.3, 3.4, 5.6, 2.4],    # Iris-virginica    | 0.00357163      0.2487635         0.8751316
    [1.3, 6.4, 3.6, 1.4],    # ?                 | 0.99455255      0.00170299        0.0077121
    ])
x = scaler.transform(x)
x = np.insert(x, 0, 1, axis=1)

print('Cost\n', calc_h(x, t))

plt.plot(j_ind, j_val)
plt.show()
