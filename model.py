# https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
original_X = data.iloc[:, 0]
Y = data.iloc[:, 1]
# plt.scatter(X, Y)
# plt.show()

# Normalization
X = (original_X - min(original_X))/(max(original_X) - min(original_X))

# Building the model
m = 0
c = 0

L = 0.1  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

emes = [0]
ces = [0]

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

    mp, cp = np.meshgrid(np.linspace(-6000, 2000, int(n)), np.linspace(-1000, 9000, int(n)))
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
    ax.set_ylabel('c')
    ax.set_zlabel('error')
    zetas = [error(m, c) for m, c in zip(emes, ces)]
    ax.plot(emes, ces, zetas, color='red', linewidth=2)
    plt.show()

def least_squares(x, y):
    average_x = sum(x) / len(x)
    average_y = sum(y) / len(y)

    theta1 = sum((x - average_x) * (y - average_y)) / sum((x - average_x)**2)
    theta0 = average_y - theta1 * average_x;
    return (theta0, theta1)

# loss_fun()

t0, t1 = least_squares(X, Y)
print(f't0: {t0}, t1: {t1}')
nt0, nt1 = least_squares(original_X, Y)
print(f'non-normalize: t0: {nt0}, t1: {nt1}')
rt1 = t1 / (max(original_X) - min(original_X))
rt0 = t0 - (min(original_X) * rt1)
print(f't1 / ({max(original_X)} - {min(original_X)}) = {rt1}')
print(f't0 - ({min(original_X)} * rt0) = {rt0}')

# Dis-normalization
dis_m = m / (max(original_X) - min(original_X))
dis_c = c - (min(original_X) * dis_m)
print(dis_m, dis_c)

# Making predictions
# Y_pred = m*X + c
Y_pred = dis_m*original_X + dis_c

# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='red')  # regression line
plt.scatter(original_X, Y)
plt.plot(original_X, Y_pred, color='red')  # regression line
plt.show()
