# https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
# plt.scatter(X, Y)
# plt.show()

# Building the model
m = 0
c = 0

L = 0.000000000001  # The learning Rate
epochs = 200  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
# for i in range(epochs): 
#     Y_pred = m*X + c  # The current predicted value of Y
#     D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c
#     loss = (-1/n) * (sum(Y - Y_pred))**2
#     # print (m, c)
#     print(loss)
    
# print (m, c)

def loss_fun():
    mp = np.outer(np.linspace(-1000000000, 1000000000, int(n)), np.ones(int(n)))
    cp = mp.copy().T
    xx = np.outer(X, np.ones(int(n)))
    yy = np.outer(Y, np.ones(int(n)))
    y_pre = mp*xx + cp
    zp = (-1/n) * (sum(yy - y_pre))**2

    print(cp.shape)
    print(yy.shape)
    print(y_pre.shape)
    print(zp.shape)

    ax = plt.axes(projection='3d')
    ax.plot_surface(mp, cp, zp, cmap='viridis', edgecolor='green')
    # ax.plot3D(mp, cp, zp, 'green')
    plt.show()

loss_fun()

# Making predictions
# Y_pred = m*X + c

# plt.scatter(X, Y) 
# plt.plot(X, Y_pred, color='red')  # regression line
# plt.show()