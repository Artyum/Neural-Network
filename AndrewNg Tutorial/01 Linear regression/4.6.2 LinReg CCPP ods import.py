# The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load.
# Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net
# hourly electrical energy output (EP)  of the plant.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from pandas_ods_reader import read_ods

rawdata = read_ods('../DataSet/Combined Cycle Power Plant/Folds5x2_pp.ods', 1)

# print(rawdata.shape)
# print(rawdata.head(3))
# print(rawdata.describe)
# print(rawdata.columns.values)
# exit()

data = rawdata.astype(float)
# print(data.head(3))
# exit()

x_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1:]
# print('x_train\n', x_train.head(3))
# print('y_train\n', y_train.head(3))
# exit()

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_train = np.array(x_train)
x_train = np.insert(x_train, 0, 1, axis=1)    # Inserting "bias column = 1
y_train = np.array(y_train)

# print('x_train\n', x_train[:3])
# print('y_train\n', y_train[:3])
# exit()

def calc_h(x, t):
    h = np.dot(x, t)
    return h

x = x_train[:9000]
y = y_train[:9000]

print('Training set =', x.shape[0])
# exit()

# Normal Equation
t = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

# print('Theta:\n', t)

# dict = ['AT', 'V', 'AP', 'RH', 'PE']
x = np.array([                            # TRUE    |  EST.
    [13.79, 42.07, 1018.27, 88.06],       # 462.25  |  466.74
    [15.12, 48.92, 1011.8, 72.93],        # 462.59  |  464.51
    [33.41, 77.95, 1010.3, 59.72],        # 432.9   |  423.63
    [15.99, 43.34, 1014.2, 78.66],        # 465.96  |  463.34
    [17.65, 59.87, 1018.58, 94.65],       # 450.93  |  453.96
    [23.68, 51.3, 1011.86, 71.24],        # 451.67  |  447.32
    ])

x = scaler.transform(x)
x = np.insert(x, 0, 1, axis=1)

# print('x\n', x)
print('Cost\n', calc_h(x, t))
