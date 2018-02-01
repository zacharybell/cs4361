import math
import numpy as np


def count_occur(a: list):
    occur = {}
    for e in a:
        if e in occur:
            occur[e] += 1
        else:
            occur[e] = 1
    return occur


def entropy(a: list):
    a_dict = count_occur(a)
    total = len(a)
    return sum([-(val/total) * math.log(val/total, 2) for val in a_dict.values()])


def convert(x: list, y: list) -> dict:
    x_dict = {}

    for i in range(len(x)):
        if x[i] in x_dict:
            x_dict[x[i]].append(y[i])
        else:
            x_dict[x[i]] = [y[i]]
    return x_dict


def gain(x: np.ndarray, y: list, index: int):
    features = x[:, index]
    feat_dict = convert(features, y)

    child_entropy = 0
    for c in feat_dict.values():
        child_entropy += (len(c) / len(y)) * entropy(c)

    return entropy(y) - child_entropy


X = np.array([
    [0, 1, 0],
    [1, 2, 0],
    [2, 0, 2],
    [1, 1, 1],
    [0, 2, 1]
])

Y = [0, 0, 1, 1, 1]

print("a0: ", gain(X, Y, 0))
print("a1: ", gain(X, Y, 1))
print("a2: ", gain(X, Y, 2))


