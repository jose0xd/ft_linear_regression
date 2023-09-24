import numpy as np
import pandas as pd

# https://www.symmetric.es/blog/regresion-lineal-con-gradiente-descendente/
def gradient_descent_old(x, y, theta0 = 0, theta1 = 0, iterations = 10000, alpha = 0.01, epsilon = 0.01):
     
    def cost_function(x, y):
        return lambda theta0, theta1: np.sum((theta0 + theta1 * x - y) ** 2) / len(x)
     
    def derivative_theta_0(x, y):
        return lambda theta0, theta1: 2/len(x) * np.sum(theta0 + theta1 * x - y)
 
    def derivative_theta_1(x, y):
        return lambda theta0, theta1: 2/len(x) * np.sum((theta0 + theta1 * x - y) * x)
     
    J = cost_function(x,y)
    J0 = derivative_theta_0(x,y)
    J1 = derivative_theta_1(x,y)
     
    convergence = False
    for i in range(0,iterations):
        cost = J(theta0,theta1)
        Jp0 = J0(theta0,theta1)
        Jp1 = J1(theta0,theta1)
 
        theta0 = theta0 - alpha * Jp0
        theta1 = theta1 - alpha * Jp1
        print(f"Theta0: {theta0}, Theta1: {theta1}")
 
        cost_new = J(theta0,theta1) 
        convergence = np.abs(cost_new - cost) < epsilon
        cost = cost_new
 
        if convergence == True:
            print("Convergence FOUND!")
            print("Theta0: " + str(theta0))
            print("Theta1: " + str(theta1))
            print(str(i) + " iterations")
            print("Cost: " + str(cost))
            break
     
    if convergence == True:
        return theta0,theta1
    else:
        return 0,0

# https://www.youtube.com/watch?v=VmbA0pi2cRQ
def loss_function(a, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].km
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].km
        y = points.iloc[i].price

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b



data = pd.read_csv("data.csv")
x = data["km"]
y = data["price"]
th0, th1 = gradient_descent_old(x, y, alpha=0.00000000001, iterations=1000, epsilon=0)
print(f"th1: {th1}, th0: {th0}")

m = 0
b = 0
L = 0.00000000001
epochs = 1000

for i in range(epochs):
    # if i % 50 == 0:
    #     print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print(f"m: {m}, b: {b}")