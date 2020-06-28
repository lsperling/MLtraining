# Load library
import matplotlib.pyplot as plt
import numpy as np

# import data
with open('machine-learning-ex1/ex1/ex1data1.txt', 'r') as data_file:
    data = data_file.read()
data = data.split()
population, profit = [], []
for ts in data:
    ts = ts.split(',')
    population.append(float(ts[0]))
    profit.append(float(ts[1]))


# Preparing the data
pop_vec = np.array(population)
x0 = np.ones(pop_vec.size).reshape((pop_vec.size),1)
pop_vec = pop_vec.reshape((pop_vec.size),1)
x = np.concatenate((x0,pop_vec), axis=1)
y = np.array(profit).reshape(len(profit),1)


# Cost function J
def compute_cost(x, y, theta):
    distance = (np.dot(x, theta) - y) ** 2
    return sum(distance) / (2 * len(y))


# Gradient Descent algorithm
def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    theta0_vals = list(theta[0])
    theta1_vals = list(theta[1])
    for iter in range(iterations):
        theta[0], theta[1] = (theta[0] - (alpha / m) * sum((np.dot(x, theta) - y) * x[:, 0].reshape(m,1)),
                              theta[1] - (alpha / m) * sum((np.dot(x, theta) - y) * x[:, 1].reshape(m,1)))
        theta0_vals.extend(theta[0])
        theta1_vals.extend(theta[1])
    return [theta, theta0_vals, theta1_vals]


# Gradient Descent algorithm
iterations = 1500
alpha = 0.01
theta_vec = np.array([[0.0], [0.0]])
theta_vec, theta0_vals, theta1_vals = gradient_descent(x, y, theta_vec, alpha, iterations)
print(theta_vec)


# Calculate normal equation
calculated_theta = np.linalg.inv(x.T @ x) @ x.T @ y
print(calculated_theta)

# Plot
plt.plot(population, profit, 'g.', x[:, 1], x @ theta_vec, 'r-', x[:, 1], x @ calculated_theta, 'b-')
plt.show()