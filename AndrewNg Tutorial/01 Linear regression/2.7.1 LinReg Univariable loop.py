# https://www.youtube.com/watch?v=GtSf2T6Co80&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def calc_h(t0, t1, x):
    return t0 + t1*x

def calc_cost(h, y):
    return (h - y)**2

t0 = np.random.random()*2-1
t1 = np.random.random()*2-1
alpha = 0.02

iter = np.array([])
j_iter = np.array([])

x_value = [x for x in range(25)]

for e in range(10001):
    # Calculating cost
    cost_sum = 0
    for i in range(m):
        h = calc_h(t0, t1, x[i])
        c = calc_cost(h, y[i])
        cost_sum = cost_sum + c

    j = 1/(2*m) * cost_sum
    iter = np.append(iter, e)
    j_iter = np.append(j_iter, j)

    if e%100 == 0:
        print('{0} j = {1}'.format(e, j))
        # Regression line
        y_value = [t0 + t1 * x for x in x_value]
        ax[0].plot(x_value, y_value, color="y")

    sum_dt0 = 0
    sum_dt1 = 0

    # Calculating derivatives
    for i in range(m):
        h = calc_h(t0, t1, x[i])
        dt0 = h - y[i]
        dt1 = dt0 * x[i]
        sum_dt0 = sum_dt0 + dt0
        sum_dt1 = sum_dt1 + dt1

    t0 = t0 - alpha/m*sum_dt0
    t1 = t1 - alpha/m*sum_dt1

# Regression line
y_value = [t0 + t1 * x for x in x_value]
ax[0].plot(x_value, y_value, color="g")

# Cost function
ax[1].plot(iter, j_iter)

# Test
ax[0].scatter(5, calc_h(t0, t1, 5), color="y")
ax[0].scatter(10, calc_h(t0, t1, 10), color="y")
ax[0].scatter(15, calc_h(t0, t1, 15), color="y")
ax[0].scatter(20, calc_h(t0, t1, 20), color="y")

plt.show()
