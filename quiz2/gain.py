import math
import numpy as np


def count_occur(a: list):
    """
    Counts the number of occurrences of each unique value.

    Example: count_occur([0, 1, 0, 0]) returns {0: 3, 1: 1}

    :param a: a list of values
    :return: a dict of unique values as keys and their counts as values
    """
    occur = {}
    for e in a:
        if e in occur:
            occur[e] += 1
        else:
            occur[e] = 1
    return occur


def entropy(a: list) -> float:
    """
    Computes entropy based on the total and number of unique values in a list.

    :param a: a list of labels
    :return: the entropy
    """

    a_dict = count_occur(a)
    total = len(a)
    return sum([-(val/total) * math.log(val/total, 2) for val in a_dict.values()])


def convert(x: list, y: list) -> dict:
    """
    Converts features and labels into a converted format that can be used to get
    counts and compute entropy.

    Example: x = [0, 1, 2, 2] and y = [1, 0, 0, 1] would become {0: [1], 1: [0], 2: [0, 1]}

    This is useful for returning all labels directly associated with a feature.

    :param x: features
    :param y: labels
    :return: a converted combination of x and y
    """

    x_dict = {}

    for i in range(len(x)):
        if x[i] in x_dict:
            x_dict[x[i]].append(y[i])
        else:
            x_dict[x[i]] = [y[i]]
    return x_dict


def gain(x: np.ndarray, y: list, index: int):
    """
    Computes the gain from using a particular feature in a set of possible features. This
    value is useful in deciding which value to use to split data in a decision tree.

    :param x: a matrix (numpy array) of features and their possible values
    :param y: labels
    :param index: the index of the feature under consideration
    :return: the gain
    """

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


