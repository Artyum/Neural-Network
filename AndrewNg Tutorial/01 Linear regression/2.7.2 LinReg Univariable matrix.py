# https://www.youtube.com/watch?v=GtSf2T6Co80&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Lineral regression with one variable

data = pd.read_csv('../DataSet/Linear regression/ex1data1.txt', header=None)
data = data.astype(float)

# print(data.head())
# print(data.describe())

fig, ax = plt.subplots(2)
ax[0].scatter(data[0], data[1])

# plt.scatter(data[0], data[1])
# plt.xticks(np.arange(5, 30, step=5))
# plt.yticks(np.arange(-5, 30, step=5))

# plt.xlabel("Population of City (10,000s)")
# plt.ylabel("Profit ($10,000")
# plt.title("Profit Vs Population")
# ax[0].set_xlabel('Population of City (10,000s)')
ax[0].set_title('Profit vs Population')
ax[1].set_title('Cost function')

x = np.array(data.iloc[:, :1])
y = np.array(data.iloc[:, 1:])
m = len(x)

# print(x)
# print(y)
# exit()

def calc_h(t, x):
    # x = np.insert(x, 0, 1, axis=1)
    # r = np.dot(x, t)

    r = t[0] + t[1]*x

    # print(r)
    # exit()

    return r

def calc_cost(h, y):
    s = (h - y)**2
    return s.sum()

t0 = np.random.random()*2-1
t1 = np.random.random()*2-1

t = np.array([[t0], [t1]])
alpha = 0.01

iter = np.array([])
j_iter = np.array([])

x_value = [x for x in range(25)]
# print(x_value)
# exit

pre_j = 999
for e in range(10001):
    # Calculating cost
    h = calc_h(t, x)
    c = calc_cost(h, y)

    j = 1/(2*m) * c
    iter = np.append(iter, e)
    j_iter = np.append(j_iter, j)

    # Print current cost function
    if e%500 == 0:
        print('Epoch: {0} | j = {1}'.format(e, j))
        # Regression line
        y_value = [calc_h(t, x) for x in x_value]
        ax[0].plot(x_value, y_value, color="y")

    if math.fabs(pre_j-j) < 0.0000000001:
        print('Epoch: {0} | j = {1}'.format(e, j))
        break
    pre_j = j

    # Calculating derivatives
    dt0 = h - y
    dt1 = dt0 * x

    t[0] = t[0] - alpha/m*dt0.sum()
    t[1] = t[1] - alpha/m*dt1.sum()

# Weights
print('t0 =', t[0])
print('t1 =', t[1])

# Regression line
y_value = [t[0] + t[1] * x for x in x_value]
ax[0].plot(x_value, y_value, color="g")

# Cost function
ax[1].plot(iter, j_iter)

# Test
ax[0].scatter(5, calc_h(t, 5), color="y")
ax[0].scatter(10, calc_h(t, 10), color="y")
ax[0].scatter(15, calc_h(t, 15), color="y")
ax[0].scatter(20, calc_h(t, 20), color="y")

plt.show()
