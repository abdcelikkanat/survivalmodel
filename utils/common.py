import os
import re
import math
import random
import torch
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle

# Path definitions
BASE_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

# Constants
EPS = 1e-6
INF = 1e+6
PI = math.pi
LOG2PI = math.log(2*PI)

# Constants for the erfi function
_ERFI_P = .3275911
_ERFI_A1 = .254829592
_ERFI_A2 = -.284496736
_ERFI_A3 = 1.421413741
_ERFI_A4 = -1.453152027
_ERFI_A5 = 1.061405429


def str2int(text):

    return int(sum(map(ord, text)) % 1e6)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pair_iter(n, directed=False):

    if directed:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                else:
                    yield i, j

    else:
        for i in range(n):
            for j in range(i+1, n):
                yield i, j

# Approximate the erf function for the real values
def erfi_approx(z: torch.Tensor):
    '''
    Approximate the erf function with maximum error of 1.5e-7, 
    Source: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables by Abramowitz M. and Stegun I.A.
            Equation 7.1.26
    '''
    
    z_ = 1j * z
    t = 1.0 / (1.0 + _ERFI_P*z_)

    return ( 1 - (t*(_ERFI_A1 + t*(_ERFI_A2 + t*(_ERFI_A3 + t*(_ERFI_A4 + t*_ERFI_A5))))) * torch.exp(-z_**2) ).imag


def pairIdx2flatIdx(i, j, n, directed=False):

    if not directed:

        return (n-1) * i - (i*(i+1)//2) + (j-1)

    else:

        return i*n + j - i - 1*(j>i)

# A method converting linear index to pair index
def linearIdx2matIdx(idx, n, dtype=torch.long, directed=False):

    if directed:

        row_idx = idx // (n-1)
        col_idx = idx % (n-1)
        col_idx[col_idx >= row_idx] += 1

        return torch.vstack((row_idx, col_idx)).to(torch.long)

    else:

        r = torch.ceil(n - 1 - 0.5 - torch.sqrt(n ** 2 - n - 2 * idx - 1.75)).type(dtype)
        c = idx - r * n + ((r + 1) * (r + 2)) // 2

        return torch.vstack((r.type(dtype), c.type(dtype)))


# def plot_events(num_of_nodes, samples, labels, title=""):

#     def node_pairs(num_of_nodes):
#         for idx1 in range(num_of_nodes):
#             for idx2 in range(idx1 + 1, num_of_nodes):
#                 yield idx1, idx2
#     pair2idx = {pair: idx for idx, pair in enumerate(node_pairs(num_of_nodes))}

#     samples, labels = shuffle(samples, labels)

#     plt.figure(figsize=(18, 10))
#     x = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}
#     y = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}
#     c = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}

#     for sample, label in zip(samples, labels):

#         idx1, idx2, e = int(sample[0]), int(sample[1]), float(sample[2])

#         x[idx1][idx2].append(e)
#         y[idx1][idx2].append(pair2idx[(idx1, idx2)])
#         c[idx1][idx2].append(label)

#     colors = ['.r', 'xk']
#     for idx1, idx2 in node_pairs(num_of_nodes):

#         for idx3 in range(len(x[idx1][idx2])):
#             # if colors[c[idx1][idx2][idx3]] != '.r':
#             plt.plot(x[idx1][idx2][idx3], y[idx1][idx2][idx3], colors[c[idx1][idx2][idx3]])

#     plt.grid(axis='x')
#     plt.title(title)
#     plt.show()

    
