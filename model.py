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
c = 9000

L = 0.00000000001  # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

emes = [0]
ces = [9000]

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    loss = (-1/n) * (sum(Y - Y_pred))**2
    # print (m, c)
    print(f'm: {m}, c: {c}, loss: {loss}')
    emes.append(m)
    ces.append(c)
    
print (m, c)

def loss_fun():
    def error(m, c):
        totalError = 0
        for i in range(0, int(n)):
            totalError += (Y[i] - (m * X[i] + c)) ** 2
        return totalError / n

    mp, cp = np.meshgrid(np.linspace(-1000, 1000, int(n)), np.linspace(-1000, 1000, int(n)))
    # xx = np.outer(X, np.ones(int(n)))
    # yy = np.outer(Y, np.ones(int(n)))
    # y_pre = mp*X + cp
    # zp = (-1/n) * (sum(Y - y_pre))**2
    zp = np.array([error(m, c) for m, c in zip(np.ravel(mp), np.ravel(cp))])
    zp = zp.reshape(mp.shape)

    ax = plt.axes(projection='3d')
    ax.plot_surface(mp, cp, zp, cmap='viridis', edgecolor='green', alpha=0.5)
    # ax.plot3D(mp, cp, zp, 'green')
    ax.set_xlabel('m')
    ax.set_ylabel('b')
    ax.set_zlabel('error')
    zetas = [error(m, c) for m, c in zip(emes, ces)]
    ax.plot(emes, ces, zetas, color='red', linewidth=2)
    plt.show()

loss_fun()

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot(X, Y_pred, color='red')  # regression line
plt.show()