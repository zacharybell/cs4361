from sklearn.preprocessing import minmax_scale

import numpy as np

test = np.array(["AA", "BB", "AB", "BB"])

# 1
def string_to_int(x):
    unq = np.unique(x)
    int_array = np.zeros(len(x))

    for i, val in enumerate(x):
        idx, = np.where(unq == val)
        int_array[i] = int(idx)

    return int_array

# 2
def one_hot(x):
    n_vals = len(set(x))

    oh = np.empty([len(x), n_vals])
    for i, val in enumerate(x):
        tmp = np.zeros(n_vals)
        tmp[int(val)] = 1
        oh[i] = tmp

    return oh

# 3
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

def bin(x, rng, n):
    x_norm = normalize(x)
    rng_n = normalize(rng)
    return np.digitize(x, np.linespace(min(rng_n), max(rng_n), num=n))
