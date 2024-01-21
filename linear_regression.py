import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)


class Linear_Regression:
    def __init__(self, data_file='data.csv', learning_rate=0.5, iterations=200):
        try:
            data = pd.read_csv(data_file)
        except:
            print(f'Cannot open file {data_file}')
            sys.exit()

        self.original_X = data.iloc[:, 0]
        self.Y = data.iloc[:, 1]
        # Normalization
        self.X = (self.original_X - min(self.original_X)) \
            / (max(self.original_X) - min(self.original_X))

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = [0, 0]

        self.costs = []
        self.history = [self.theta.copy()]

    def least_squares(self):
        '''Using the least-squares approach: a line that minimizes the sum of squared residuals.'''
        average_x = sum(self.original_X) / len(self.original_X)
        average_y = sum(self.Y) / len(self.Y)

        theta1 = sum((self.original_X - average_x) * (self.Y - average_y)) / sum((self.original_X - average_x)**2)
        theta0 = average_y - theta1 * average_x
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
        print('  Real values by dis-normalization of data:')
        print(f'theta0: {self.theta[0]}, theta1: {self.theta[1]}')

    def get_current_accuracy(self, Y_pred):
        return 1 - sum(abs(Y_pred - self.Y) / self.Y) / len(Y_pred)

    def plot_data(self, X, Y, fig, line=None):
        plt.figure(fig)
        plt.scatter(X, Y, color='blue')
        if line is not None:
            plt.plot(X, line, color='red')
        plt.show()

    def plot_cost_evolution(self, start=0):
        plt.figure('Cost evolution')
        plt.plot(range(start, self.iterations), self.costs[start:], color='green')
        plt.show()

    def plot_loss_function(self):
        m = len(self.Y)

        def error(t0, t1):
            return np.sum((self.Y - (t1 * self.X + t0)) ** 2) / (2*m)

        def calculate_borders(t0_data, t1_data):
            t0min = min(t0_data)
            t0max = max(t0_data)
            t1min = min(t1_data)
            t1max = max(t1_data)
            xmin = int(t0min - (0.1 * t0max - t0min))
            xmax = int(t0max + (0.1 * t0max - t0min))
            ymin = int(t1min - (0.1 * t1max - t1min))
            ymax = int(t1max + (0.1 * t1max - t1min))
            return xmin, xmax, ymin, ymax

        t0_data, t1_data = list(zip(*self.history))
        xmin, xmax, ymin, ymax = calculate_borders(t0_data, t1_data)
        xp, yp = np.meshgrid(np.linspace(xmin, xmax, m), np.linspace(ymin, ymax, m))
        zp = np.array([error(t0, t1) for t0, t1 in zip(np.ravel(xp), np.ravel(yp))])
        zp = zp.reshape(xp.shape)

        errors = [error(t0, t1) for t0, t1 in self.history]

        ax = plt.axes(projection='3d')
        ax.set_xlabel('theta0')
        ax.set_ylabel('theta1')
        ax.set_zlabel('error')
        # Plot loss function
        ax.plot_surface(xp, yp, zp, cmap='viridis', edgecolor='green', alpha=0.5)
        # Plot evolution of error
        ax.plot(t0_data, t1_data, errors, color='red', linewidth=2)
        plt.show()

    def make_predictions(self):
        while True:
            value = input('Introduce a value to predict the output or "quit": ')
            if value == 'quit' or value == 'q':
                break
            if not value.isdigit():
                print('It should be a number')
                continue
            value = float(value)
            print(f'From {value}, model predict: {self.predict(value)}')


def main():
    # Parse args
    parser = argparse.ArgumentParser(description='Linear regression model.')
    parser.add_argument('datafile', help='file name of the csv data',
                        nargs='?', default='data.csv')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.5)
    parser.add_argument('-i', '--iterations', type=int, default=200)
    parser.add_argument('-e', '--show_error_function', action='store_true', default=False,
                        help='Plot the evolution of the error and the loss function.')
    parser.add_argument('-s', '--least_squares', action='store_true', default=False,
                        help='Use the method of least squares to calculate linear regression.')
    parser.add_argument('-p', '--predict', action='store_true', default=False,
                        help='Use the model to predict values.')
    args = parser.parse_args()

    # Create model
    model = Linear_Regression(args.datafile, args.learning_rate, args.iterations)

    if args.least_squares:
        theta0, theta1 = model.least_squares()
        print(f'theta0: {theta0}, theta1: {theta1}')
        # Plot line
        Y_pred = theta0 + (theta1 * model.original_X)
        model.plot_data(model.original_X, model.Y, 'Plotting data', Y_pred)
        print(f'Current accuracy: {model.get_current_accuracy(Y_pred)}')
        return

    if args.predict:
        model.train_model(logs=False)
        model.make_predictions()
        return

    # Plot data
    model.plot_data(model.original_X, model.Y, 'Plotting data')

    model.train_model(save_history=args.show_error_function)

    # Plot line
    Y_pred = model.predict(model.original_X)
    model.plot_data(model.original_X, model.Y, 'Plotting data', Y_pred)
    print(f'Current accuracy: {model.get_current_accuracy(Y_pred)}')

    if args.show_error_function:
        model.plot_cost_evolution(start=0)
        model.plot_loss_function()


if __name__ == '__main__':
    main()
