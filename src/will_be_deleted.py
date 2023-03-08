import matplotlib.pyplot as plt
import torch
import random


'''
This is a uniform sampling strategy to select entries from the upper triangular part of a network
Note that the sampled entries might overlap with networks' links
'''

# Define a method mapping (i,j) pairs to indices starting from 0 to (N-1)*N/2  - 1
def pairIdx2flatIdx(i, j, n):

    return (n-1) * i - int(i*(i+1)/2) + (j-1)

# A method converting linear index to pair index
def linearIdx2pairIdx(idx, n):

    r = torch.ceil(n - 1 - 0.5 - torch.sqrt(n ** 2 - n - 2 * idx - 1.75)).type(torch.int)
    c = idx - r * n + ((r + 1) * (r + 2)) // 2

    return torch.vstack((r.type(torch.int), c.type(torch.int))).T

# Define a seed value
seed = 19
nodes_num = 16
threshold = 0.8

# Set the seed for reprooducibility
random.seed(seed)
torch.manual_seed(seed)

# Define an adjacency matrix
A = torch.asarray(torch.rand(nodes_num, nodes_num) > threshold, dtype=int)

# Take the upper triangular part
A = torch.triu(A, 1)

# Get the node degrees
node_degrees = torch.sum(A, dim=1)

# Define the number of entries in the upper triangular part 
m = (nodes_num - 1)*nodes_num // 2

# Define the total number of negative instances
k = sum(node_degrees)

# Sample all the negative instances without replacement
perm = torch.randperm(m)
chosen_samples_flatidx = perm[:k]

# Sort the chosen samples
chosen_samples_flatidx, _ = torch.sort(chosen_samples_flatidx)

# Convert the linear index to pair indices which is a matrix of size 2 x k
pair_indices = linearIdx2pairIdx(chosen_samples_flatidx, n=nodes_num)

# We do not need to care about the row indices of the pairs anymore so we can simply set them to the corresponding row indices
temp_idx = torch.repeat_interleave(torch.arange(nodes_num), repeats=node_degrees)
pair_indices[:, 0] = temp_idx.type(torch.int)

# Choose the samples with respect to the node degrees (if neeed)
split_sections = torch.split(pair_indices, node_degrees.tolist(), dim=0)

# Construct the sample matrix
S = torch.zeros_like(A)
for pairs in split_sections:
    for pair in pairs:
        i, j = pair
        S[i, j] = 1


# Plot the adjacency and sampled entries
f, ax = plt.subplots(1,2)
ax[0].title.set_text('Adjacency Matrix')
ax[0].imshow(A, cmap='gray_r')
ax[1].title.set_text('Sampled entries')
ax[1].imshow(S, cmap='Reds')
plt.show()
