import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

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
x_train = np.insert(x_train, 0, 1, axis=1)    # Inserting "bias column = 1
y_train = np.array(y_train)

# print('x_train\n', x_train[:3])
# print('y_train\n', y_train[:3])
# exit()

features_num = 8     # Bias + 7

# np.random.seed(5)
t = np.random.rand(features_num, 1)*2-1

def calc_h(x, t):
    # print(x)
    # print(t)
    # exit()
    h = np.dot(x, t)
    # print(h.shape)
    # exit()

    # Hypotheses for each test case
    return h

def calc_cost(h, y):
    s = (h - y)**2
    # print(s)
    # print(h.shape)
    # exit()
    m = h.shape[0]   # Number of test samples
    return 1/(2*m) * s.sum()

x = x_train
y = y_train
alpha = 0.1
m = x.shape[0]
print('training_set =', m)

# print(x)
# print(y)
# print(t)
# exit()

j_ind = np.array([])
j_val = np.array([])

j_pre = 999
for i in range(500001):
    # Hypotheses
    h = calc_h(x, t)

    # Cost
    j = calc_cost(h, y)
    if i % 1000 == 0:
        print('Epoch: {0}  |  Cost: {1}'.format(i, j))

    # Plot of the cost function
    if i > 100 and i % 100 == 0:
        j_ind = np.append(j_ind, i)
        j_val = np.append(j_val, j)

    # Exit when the end of training
    if math.fabs(j_pre - j) < 0.00001:
        print('Epoch: {0}  |  Cost: {1}'.format(i, j))
        break
    j_pre = j

    e = h - y  # (1460 x 1)
    dt = e * x
    # print(dt.shape)
    dt = dt.sum(axis=0)  # Sum by columns
    dt = dt.reshape(-1, 1)
    # print(dt.shape)
    # print(dt)
    # exit()
    # print(t)
    # exit()
    t = t - alpha/m * dt
    # print(t)
    # exit()

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
