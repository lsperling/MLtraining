# Load library
import matplotlib.pyplot as plt
import numpy as np


# Import data
def training_set_import():
    with open('machine-learning-ex1/ex1/ex1data2.txt', 'r') as data_file:
        data = data_file.readlines()
    data = [set.strip().split(',') for set in data]
    data = [[float(item) for item in set] for set in data]
    data_array = np.array(data)
    X = data_array[:, :-1]
    y = data_array[:, -1:]
    return X, y, len(y)


# Normalizing features
def featureNormalize(X, m):
    n = X.shape[1]
    features = np.hsplit(X, n)
    first_feature = True
    mu, sigma = [], []
    for feature in features:
        feature_mu, feature_sigma = feature.mean(), feature.std()
        mu.append(feature_mu)
        sigma.append(feature_sigma)
        if first_feature:
            norm_features = (feature - feature_mu) / feature_sigma
            first_feature = False
        else:
            norm_features = np.concatenate((norm_features, (feature - feature_mu) / feature_sigma), axis=1)
    return norm_features, mu, sigma


def featureRecalculate(normX, mu, sigma):
    n = normX.shape[1]
    norm_features = np.hsplit(normX, n)
    first_feature = True
    for feature_index in range(n):
        if first_feature:
            features = norm_features[feature_index] * sigma[feature_index] + mu[feature_index]
            first_feature = False
        else:
            features = np.concatenate((features, norm_features[feature_index] * sigma[feature_index] + mu[feature_index]), axis=1)
    return features


def add_x0(X):
    return np.concatenate((np.ones(X.shape[0]).reshape(X.shape[0], 1), X), axis=1)


def numerical_theta(X, y):
    numerical_theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return numerical_theta


def plot_LR(X, y, X_mod, theta):
    plt.plot(X, y, 'g.', X[:, 1:], X_mod @ theta, 'b-')
    plt.show()


X, y, m = training_set_import()
normX, mu, sigma = featureNormalize(X, m)
#featureRecalculate(normX, mu, sigma)
X_mod = add_x0(X)
theta = numerical_theta(X_mod, y)
print(X_mod.shape)
total = 0
for index in range(X_mod.shape[0]):
    total += X_mod[index] @ theta - y[index]
print(total)
#plot_LR(X, y, X_mod, theta)

