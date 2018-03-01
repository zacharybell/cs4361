import numpy as np
import sys

X_test = np.array([[1, 2, 3],[1, 3, 4],[1, 4, -5]])


# (a)
def min_max_normalize(X):
    return ((X - np.amin(X, axis=0)) /
            (np.amax(X, axis=0) - np.amin(X, axis=0) + sys.float_info.epsilon))


# (b)
def standardize(X):
    return ((X - np.average(X, axis=0)) /
            (np.std(X, axis=0) + sys.float_info.epsilon))


# (c)
def binary(X):
    mean = np.average(X, axis=0)
    return np.where(X > mean, 1, 0)


# (d)
def p_coef(x, y):
#     return ((x - np.mean(x)) * (y - np.mean(y)) / ((np.std(x) * np.std(y))) + sys.float_info.epsilon)
#
# # print(p_coef(X_test[:,0], X_test[:,1]))
# print(np.std(X_test[:, 0]))
    pass


# (e)
def augment_product(X, i, j):
    return np.concatenate((X, X[:, i:i+1] * X[:, j:j+1]), axis=1)


# (f)
def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))

def nearest_neighbor(X, y, x, weight_func=euclidean_dist):
    if X.shape[0] is not y.shape[0]:
        raise Exception('arguments must have same row dimension', X.shape, y.shape)

    min_val = sys.float_info.max
    min_index = -1

    for i, r in enumerate(X):
        print(r)
        tmp = weight_func(r, x)
        if tmp < min_val:
            min_index = i
            min_val = tmp

    return y[min_index] if min_index is not -1 else None


print(nearest_neighbor(X_test, np.array([1, 0]), np.array([])))
