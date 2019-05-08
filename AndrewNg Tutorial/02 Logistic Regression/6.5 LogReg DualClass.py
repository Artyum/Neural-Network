import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

def calc_h(x, t):
    # print(x)
    # print(t)
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

rawdata = pd.read_csv('../DataSet/Survival breast cancer/haberman.data', header=None)
# print(rawdata.descrbe())
# print(rawdata.head(5))
# exit()

# Replace values in the class column
# 1->0 = the patient survived 5 years or longer
# 2->1 = the patient died within 5 year
rawdata[3] = rawdata[3].map({1: 0, 2: 1})
# print(rawdata[3])
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
x_train = np.insert(x_train, 0, 1, axis=1)

# print(x_train[:5])
# print(y_train[:5])
# print(x_train.shape)
# print(y_train.shape)
# exit()

# np.random.seed(5)
features_num = 3
t = np.random.rand(features_num+1, 1)*2-1   # +1 for bias
# print(t)
# exit()

x = x_train
y = y_train

m = x.shape[0]
alpha = 0.3

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
    if math.fabs(j_pre - j) < 0.00000000001:
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
x = np.array([      # TRUE  | EST. [*100%]
    [37, 60, 0],    # 0     | 0.15396093
    [42, 59, 0],    # 1     | 0.16859955
    [43, 64, 3],    # 0     | 0.20379413
    [45, 67, 1],    # 1     | 0.17840405
    [37, 81, 0],    # ?     | 0.12912666
    [37, 119, 0],   # ?     | 0.0928482
    [37, 119, 3],   # ?     | 0.11738373
    [37, 119, 8],   # ?     | 0.17050516
    ])
x = scaler.transform(x)
x = np.insert(x, 0, 1, axis=1)

print('Cost\n', calc_h(x, t))

plt.plot(j_ind, j_val)
plt.show()
