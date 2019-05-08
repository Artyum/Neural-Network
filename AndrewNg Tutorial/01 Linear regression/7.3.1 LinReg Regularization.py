import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def calc_h(x, t):
    h = np.dot(x, t)
    return h

def calc_cost(h, y, r):
    s = (h - y)**2
    m = h.shape[0]   # Number of test samples
    return 1/(2*m) * (s.sum() + r)   # Regularization

rawdata = pd.read_excel('../DataSet/HousingPrices/Data.xlsx', header=0)
rawdata = shuffle(rawdata)

# print(rawdata.shape)
# print(rawdata.head(3))
# print(rawdata.describe)
# print(rawdata.columns.values)

dict = ['LotArea', 'YearBuilt', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'GarageArea', 'PoolArea', 'SalePrice']
data = rawdata[dict].astype(float)
# print(data.head(3))

x_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1:]
# print('x_train\n', x_train.head(3))
# print('y_train\n', y_train.head(3))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
# scaled_df = pd.DataFrame(scaled_df)
# print('x_train\n', x_train.head(3))
# exit()

x_train = np.array(x_train)
x_train = np.insert(x_train, 0, 1, axis=1)    # Inserting bias column = 1
y_train = np.array(y_train)

# print('x_train\n', x_train[:3])
# print('y_train\n', y_train[:3])
# exit()

features_num = 8     # Bias + 7

# np.random.seed(5)
t = np.random.rand(features_num, 1)*2-1

x = x_train
y = y_train

m = x.shape[0]
print('training_set =', m)

# print(x)
# print(y)
# print(t)
# exit()

j_ind = np.array([])
j_val = np.array([])

alpha = 0.1

# Regularization
# High lambda values flatten the results
lambda_ = 0.1

j_pre = 999
for i in range(999999):
    h = calc_h(x, t)

    # Regularization
    r = lambda_ * (t[1:]).sum()
    j = calc_cost(h, y, r)

    if i % 1000 == 0:
        print('Epoch: {0}  |  Cost: {1}'.format(i, j))

    # Plot of the cost function
    if i > 100 and i % 100 == 0:
        j_ind = np.append(j_ind, i)
        j_val = np.append(j_val, j)

    # Wyjście gdy koniec nauki
    if math.fabs(j_pre - j) < 0.00001:
        print('Epoch: {0}  |  Cost: {1}'.format(i, j))
        break
    j_pre = j

    e = h - y  # (1460 x 1)
    dt = e * x
    dt = dt.sum(axis=0)  # Sum by columns
    dt = dt.reshape(-1, 1)

    # Both calculations are correct
    # t = t - alpha * (1/m * dt + (lambda_/m)*t)
    t = t*(1-alpha * lambda_/m) - alpha/m*dt

print('Theta:\n', t)

# dict = ['LotArea', 'YearBuilt', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'GarageArea', 'PoolArea']
x = np.array([                            # TRUE    | EST.
    [14115, 1993, 1362, 1, 1, 480, 0],    # 143'000 | 215'210
    [21535, 1994, 4316, 3, 4, 832, 0],    # 755'000 | 479'052
    [15623, 1996, 4476, 3, 4, 813, 555],  # 745'000 | 466'102
    [7879, 1920, 720, 1, 2, 0, 0],        #  34'900 |  44'742
    [7879, 2000, 720, 1, 2, 0, 0],        #       ? | 107'553
    [10000, 2000, 1200, 2, 5, 300, 50],   #       ? | 126'328
    ])
x = scaler.transform(x)
x = np.insert(x, 0, 1, axis=1)

# print('x\n', x)

print('Cost\n', calc_h(x, t))
plt.plot(j_ind, j_val)
plt.show()
