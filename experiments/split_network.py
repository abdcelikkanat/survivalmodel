import os
import sys
import pickle
import networkx as nx
from argparse import ArgumentParser, RawTextHelpFormatter
import torch

import utils
from src.dataset import Dataset
from utils import set_seed, flatIdx2matIdx, matIdx2flatIdx
import numpy as np

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--edges', type=str, required=True, help='Path of the edge list file'
)
parser.add_argument(
    '--output_folder', type=str, required=True, help='Path of the output dataset folder'
)
parser.add_argument(
    '--pr', type=float, required=False, default=0.1, help='Prediction ratio'
)
parser.add_argument(
    '--mr', type=float, required=False, default=0.2, help='Masking ratio'
)
parser.add_argument(
    '--cr', type=float, required=False, default=0.1, help='Completion ratio'
)
parser.add_argument(
    '--verbose', type=bool, required=False, default=True, help='Verbose'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

# Set some parameters
edges_file = args.edges
output_folder = args.output_folder
prediction_ratio = args.pr
masking_ratio = args.mr
completion_ratio = args.cr
verbose = args.verbose
seed = args.seed

########################################################################################################################

# Set the seed value
set_seed(seed=seed)
np.random.seed(seed)

# Create the target folder
os.makedirs(output_folder)

log_file_path = os.path.join(output_folder, "log.txt")
orig_stdout = sys.stdout
f = open(log_file_path, 'w')
sys.stdout = f

# Load the dataset
dataset = Dataset()
dataset.read_edgelist(edges_file)
nodes_num = dataset.get_nodes_num()

data_dict = dataset.get_data_dict()
directed = dataset.is_directed()
signed = dataset.is_signed()
# edges, times, states = dataset.get_edges(), dataset.get_times(), dataset.get_states()

########################################################################################################################

if verbose:
    print("- The network is being divided into training and prediction sets for the future!")

# Firstly, the network will be split into two part
split_ratio = 1.0 - prediction_ratio

# Get the minimum and maximum time values
min_time, max_time = dataset.get_init_time(), dataset.get_last_time()
split_time = int(min_time + split_ratio * (max_time - min_time))


train_pairs, train_times, train_states = [], [], []
pred_pairs, pred_times, pred_states = [], [], []
for i in data_dict.keys():
    for j in data_dict[i].keys():

        train_pairs.append((i, j))
        train_times.append([])
        train_states.append([])

        pred_pairs.append((i, j))
        pred_times.append([])
        pred_states.append([])

        for time, state in data_dict[i][j]:

            if time <= split_time:
                train_times[-1].append(time)
                train_states[-1].append(state)
            else:
                pred_times[-1].append(time)
                pred_states[-1].append(state)

        # Remove the pair if it does not contain any event
        if len(train_times[-1]) == 0:
            train_pairs.pop()
            train_times.pop()
            train_states.pop()

        # If the train pair has edge, add the state for the prediction set
        else:
            pred_times[-1].append(split_time)
            pred_states[-1].append( max(zip(train_times, train_states[-1]), key=lambda value: value[0])[1] )

        # Remove the pair if it does not contain any event (train set can't contain any event but prediction set can
        if len(pred_times[-1]) == 0:
            pred_pairs.pop()
            pred_times.pop()
            pred_states.pop()

# Construct an undirected static graph from the links in the training set
train_graph = nx.DiGraph() if directed else nx.Graph()
train_graph.add_edges_from(train_pairs)

if verbose:
    print(f"\t+ Training graph has {train_graph.number_of_nodes()} nodes.")
    print(f"\t+ Training graph has {train_graph.number_of_edges()} pairs having at least one events.")
    print(f"\t+ Prediction set has {len(np.unique(np.asarray(pred_pairs)))} nodes.")
    print(f"\t+ Prediction set has {len(pred_pairs)} pairs having at least one events.")

# If there are any nodes which do not have any events during the training timeline,
# the graph must be relabeled and these nodes must be removed from the testing set as well.
newlabel = None
if len(list(nx.isolates(train_graph))) != 0:

    isolated_nodes = list(nx.isolates(train_graph))
    if verbose:
        print(f"\t\t+ Training graph has {len(isolated_nodes)} isolated nodes.")

    n, count = 0, 0
    while n < len(pred_pairs):
        i, j = pred_pairs[n]
        if i in isolated_nodes or j in isolated_nodes:
            pred_pairs.pop(n)
            pred_times.pop(n)
            pred_states.pop(n)
            count += 1
        else:
            n += 1

    # Remove the isolated nodes from the networkx graph
    train_graph.remove_nodes_from(isolated_nodes)

    if verbose:
        print(f"\t\t+ {count} pairs have been removed from the prediction set.")
        print(f"\t\t+ The prediction set has currently {len(np.unique(np.asarray(pred_pairs)))} nodes.")
        print(f"\t\t+ The prediction set has currently {len(pred_pairs)} pairs having at least one events.")

    # Set the number of nodes
    nodes_num = train_graph.number_of_nodes()

    if verbose:
        print(f"\t+ Nodes are being relabeled.")

    # Relabel nodes in the training set
    newlabel = {node: idx for idx, node in enumerate(train_graph.nodes())}
    for n, pair in enumerate(train_pairs):
        train_pairs[n] = (newlabel[pair[0]], newlabel[pair[1]])

    # Relabel nodes in the prediction set
    for n, pair in enumerate(pred_pairs):
        pred_pairs[n] = (newlabel[pair[0]], newlabel[pair[1]])

    # Relabel nodes in the networkx object
    train_graph = nx.relabel_nodes(G=train_graph, mapping=newlabel)

    if verbose:
        print(f"\t\t+ Completed.")


train_dataset = Dataset(
    nodes_num=nodes_num, directed=directed, signed=signed,
    edges=torch.repeat_interleave(
        torch.as_tensor(train_pairs, dtype=torch.long).T,
        repeats=torch.as_tensor(list(map(len, train_times)), dtype=torch.long), dim=1
    ),
    edge_times=torch.as_tensor([t for pair_times in train_times for t in pair_times], dtype=torch.long),
    edge_states=torch.as_tensor([s for pair_states in train_states for s in pair_states], dtype=torch.long),
)
min_time, max_time = train_dataset.get_init_time(), train_dataset.get_last_time()
########################################################################################################################

if verbose:
    print("- Sampling processes for the masking and completion pairs have just started!")

# Sample the masking and completion pairs
all_possible_pair_num = nodes_num * (nodes_num - 1)
if directed:
    all_possible_pair_num = all_possible_pair_num // 2

mask_size = int(all_possible_pair_num * masking_ratio)
completion_size = int(all_possible_pair_num * completion_ratio)
total_sample_size = mask_size + completion_size

# Construct pair indices
all_pos_pairs = list([(i, j) for i, j in utils.pair_iter(n=nodes_num, is_directed=directed)])
np.random.shuffle(all_pos_pairs)

# Sample node pairs such that each node in the residual has at least one event
sampled_pairs = []
for k, pair in enumerate(all_pos_pairs):
    i, j = pair

    if train_graph.has_edge(i, j):
        train_graph.remove_edge(i, j)

        if len(list(nx.isolates(train_graph))) != 0:
            train_graph.add_edge(i, j)
        else:
            sampled_pairs.append((i, j))
    else:
        sampled_pairs.append((i, j))

    if len(sampled_pairs) == total_sample_size:
        break

assert len(sampled_pairs) == total_sample_size, "Enough number of sample pairs couldn't be found!"

# Set the completion and mask pairs
mask_pairs, completion_pairs = [], []
if mask_size:
    mask_pairs = sampled_pairs[:mask_size]
if completion_size:
    completion_pairs = sampled_pairs[mask_size:]

# Set the completion and mask events
train_data_dict = train_dataset.get_data_dict()
mask_events = [[e for e, _ in train_data_dict[pair[0]][pair[1]]] if pair[0] in train_data_dict and pair[1] in train_data_dict[pair[0]] else [min_time] for pair in mask_pairs ]
mask_states = [[s for _, s in train_data_dict[pair[0]][pair[1]]] if pair[0] in train_data_dict and pair[1] in train_data_dict[pair[0]] else [0] for pair in mask_pairs]
completion_events = [[e for e, _ in train_data_dict[pair[0]][pair[1]]] if pair[0] in train_data_dict and pair[1] in train_data_dict[pair[0]] else [min_time] for pair in completion_pairs]
completion_states = [[s for _, s in train_data_dict[pair[0]][pair[1]]] if pair[0] in train_data_dict and pair[1] in train_data_dict[pair[0]] else [0] for pair in completion_pairs]

# Construct the residual pairs and events
# Since we always checked in the previous process, every node has at least one event
residual_pairs, residual_times, residual_states = train_pairs.copy(), train_times.copy(), train_states.copy()

if completion_size:
    completion_pair_indices = [int(matIdx2flatIdx(i=torch.as_tensor([pair[0]], dtype=torch.long), j=torch.as_tensor([pair[1]], dtype=torch.long), n=nodes_num, is_directed=directed)) for pair in completion_pairs]

    n = 0
    while n < len(residual_pairs):
        pair = residual_pairs[n]
        if int(matIdx2flatIdx(i=torch.as_tensor([pair[0]], dtype=torch.long), j=torch.as_tensor([pair[1]], dtype=torch.long), n=nodes_num, is_directed=directed)) in completion_pair_indices:
            residual_pairs.pop(n)
            residual_times.pop(n)
            residual_states.pop(n)
        else:
            n += 1

if verbose:
    print(f"\t+ Masking set has {mask_size} pairs.")
    mask_samples_event_pairs_num = sum([1 if len(pair_events) else 0 for pair_events in mask_events])
    print(f"\t\t+ {mask_samples_event_pairs_num} masking pairs have at least one event. ")

    print(f"\t+ Completion set has {completion_size} pairs.")
    completion_samples_event_pairs_num = sum([1 if len(pair_events) else 0 for pair_events in completion_events])
    print(f"\t\t+ {completion_samples_event_pairs_num} masking pairs have at least one event. ")

    print(f"\t+ Residual network has {len(residual_pairs)} event pairs.")

########################################################################################################################

if verbose:
    print("- The files are being written...")

# Save the training pair and events
train_path = os.path.join(output_folder, "train.edges")
triplets = [
    (pair[0], pair[1], t, s) for pair, pair_times, pair_states in zip(train_pairs, train_times, train_states)
    for t, s in zip(pair_times, pair_states)
]
with open(train_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the residual pair and events
residual_path = os.path.join(output_folder, "residual.edges")
triplets = [
    (pair[0], pair[1], t, s) for pair, pair_times, pair_states in zip(residual_pairs, residual_times, residual_states)
    for t, s in zip(pair_times, pair_states)
]
with open(residual_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the completion pair and events
completion_path = os.path.join(output_folder, "completion.edges")
triplets = [
    (pair[0], pair[1], t, s) for pair, pair_times, pair_states in zip(completion_pairs, completion_events, completion_states)
    for t, s in zip(pair_times, pair_states)
]
with open(completion_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the mask pair and events
mask_path = os.path.join(output_folder, "mask.edges")
triplets = [
    (pair[0], pair[1], t, s) for pair, pair_times, pair_states in zip(mask_pairs, mask_events, mask_states)
    for t, s in zip(pair_times, pair_states)
]
with open(mask_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")

# Save the prediction pairs
prediction_path = os.path.join(output_folder, "prediction.edges")
triplets = [
    (pair[0], pair[1], t, s) for pair, pair_times, pair_states in zip(pred_pairs, pred_times, pred_states)
    for t, s in zip(pair_times, pair_states)
]
with open(prediction_path, 'w') as f:
    for element in sorted(triplets, key=lambda x: (x[2], x[0], x[1], x[3])):
        f.write(f"{element[0]}\t{element[1]}\t{element[2]}\t{element[3]}\n")


if verbose:
    print(f"\t+ Completed.")

########################################################################################################################

sys.stdout = orig_stdout
f.close()