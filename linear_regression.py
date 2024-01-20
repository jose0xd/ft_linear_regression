import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

class Linear_Regression:
    def __init__(self, data_file='data.csv', learning_rate=0.5, iterations=200):
        data = pd.read_csv('data.csv')

        self.original_X = data.iloc[:, 0]
        self.Y = data.iloc[:, 1]
        # Normalization
        self.X = (self.original_X - min(self.original_X)) \
            / (max(self.original_X) - min(self.original_X))

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = [0, 0]

        self.costs = []
        self.history = []

    def least_squares(self):
        '''Using the least-squares approach: a line that minimizes the sum of squared residuals.'''
        average_x = sum(self.X) / len(self.X)
        average_y = sum(self.Y) / len(self.Y)

        theta1 = sum((self.X - average_x) * (self.Y - average_y)) / sum((self.X - average_x)**2)
        theta0 = average_y - theta1 * average_x;
        return (theta0, theta1)
    
    def predict(self, X):
        return self.theta[0] + (self.theta[1] * X)
    
    def compute_cost(self, Y_pred):
        m = len(self.Y)
        J = (1 / (2*m)) * (np.sum(Y_pred - self.Y)**2)
        return J

    def train_model(self, logs=True, save_history=False):
        '''Performing Gradient Descent'''
        for _ in range(self.iterations):
            Y_pred = self.predict(self.X)
            m = len(self.Y)
            self.theta[0] = self.theta[0] - (self.learning_rate * ((1/m) * np.sum(Y_pred - self.Y)))
            self.theta[1] = self.theta[1] - (self.learning_rate * ((1/m) * np.sum((Y_pred - self.Y) * self.X)))

            if logs:
                cost = self.compute_cost(Y_pred)
                self.costs.append(cost)
                print(f'theta0: {self.theta[0]}, theta1: {self.theta[1]}, cost: {cost}')

            if save_history:
                self.history.append(self.theta.copy())

        # Dis-normalization
        self.theta[1] = self.theta[1] / (max(self.original_X) - min(self.original_X))
        self.theta[0] = self.theta[0] - (min(self.original_X) * self.theta[1])
        print('Real values by dis-normalization of data')
        print(f'theta0: {self.theta[0]}, theta1: {self.theta[1]}')

        
    def get_current_accuracy(self, Y_pred):
        return 1 - sum(abs(Y_pred - self.Y) / self.Y) / len(Y_pred)

    def plot_data(self, X, Y, fig, line=None):
                plt.figure(fig)
                plt.scatter(X, Y, color='blue')
                if line is not None:
                    plt.plot(X, line, color='red')
                plt.show()

    def loss_function(self):
        m = len(self.Y)
        def error(t0, t1):
            totalError = 0
            for i in range(0, int(m)):
                totalError += (Y[i] - (t1 * X[i] + t0)) ** 2
            return totalError / m

        mp, cp = np.meshgrid(np.linspace(-6000, 2000, int(n)), np.linspace(-1000, 9000, int(m)))
        zp = np.array([error(t0, t1) for t0, t1 in zip(np.ravel(mp), np.ravel(cp))])
        zp = zp.reshape(mp.shape)

        ax = plt.axes(projection='3d')
        ax.plot_surface(mp, cp, zp, cmap='viridis', edgecolor='green', alpha=0.5)
        ax.set_xlabel('theta1')
        ax.set_ylabel('theta0')
        ax.set_zlabel('error')
        errors = [error(t0, t1) for t0, t1 in self.history]
        ax.plot(emes, ces, errors, color='red', linewidth=2)
        plt.show()


def main():
    model = Linear_Regression()

    # Plot data
    model.plot_data(model.original_X, model.Y, 'Plotting data')
    
    model.train_model()

    # Plot line
    Y_pred = model.predict(model.original_X)
    model.plot_data(model.original_X, model.Y, 'Plotting data', Y_pred)
    print(f'Current accuracy: {model.get_current_accuracy(Y_pred)}')

    # Plot cost evolution
    plt.figure('Cost evolution')
    plt.plot(range(model.iterations), model.costs, color='green')
    plt.show()


if __name__ == '__main__':
    main()
