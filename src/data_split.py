# import os
# import sys
# import pickle
# import networkx as nx
# from argparse import ArgumentParser, RawTextHelpFormatter
from src.dataset import Dataset
# from utils.common import set_seed, linearIdx2matIdx, pairIdx2flatIdx
# import numpy as np
# from sklearn.utils import shuffle
import torch

# ########################################################################################################################

# parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
# parser.add_argument(
#     '--input_folder', type=str, required=True, help='Path of the input dataset folder'
# )
# parser.add_argument(
#     '--output_folder', type=str, required=True, help='Path of the output dataset folder'
# )
# parser.add_argument(
#     '--log_file', type=str, required=False, default="", help='Path for the log file'
# )
# parser.add_argument(
#     '--pr', type=float, required=False, default=0.1, help='Prediction ratio'
# )
# parser.add_argument(
#     '--mr', type=float, required=False, default=0.2, help='Masking ratio'
# )
# parser.add_argument(
#     '--cr', type=float, required=False, default=0, help='Completion ratio'
# )
# parser.add_argument(
#     '--verbose', type=bool, required=False, default=True, help='Verbose'
# )
# parser.add_argument(
#     '--seed', type=int, required=False, default=19, help='Seed value'
# )
# args = parser.parse_args()

# ########################################################################################################################

# # Set some parameters
# input_folder = args.input_folder
# output_folder = args.output_folder
# log_file = args.log_file
# prediction_ratio = args.pr
# masking_ratio = args.mr
# completion_ratio = args.cr
# verbose = args.verbose
# seed = args.seed

# ########################################################################################################################

class DataSplit:
    def __init__(self, dataset: Dataset) -> None:

        self.__dataset = dataset
        nodes_num = dataset.get_nodes_num()

        self._adj_list = dataset.get_adj_list()

    def split_in_time(self, pairs, events, split_ratio=0.8):
        """
        Split the network in time.
        :param pairs: list of pairs
        :param events: list of events
        :param split_ratio: split ratio
        :return: train_pairs, train_events, pred_pairs, pred_events
        """

        min_time = dataset.get_init_time()
        max_time = dataset.get_last_time()
        split_time = min_time + (max_time - min_time) * split_ratio

        train_pairs, train_events = [], []
        pred_pairs, pred_events = [], []

        for i, j, t in zip(self.__dataset.get_edges().T, self.__dataset.get_times()):

        for pair, pair_events in zip(pairs, events):

            train_pairs.append(pair)
            train_events.append([])

            pred_pairs.append(pair)
            pred_events.append([])

            for e in pair_events:
                if e <= split_ratio:
                    train_events[-1].append(e)
                else:
                    pred_events[-1].append(e)

            # Remove the pair if it does not contain any event
            if len(train_events[-1]) == 0:
                train_pairs.pop()
                train_events.pop()

            # Remove the pair if it does not contain any event
            if len(pred_events[-1]) == 0:
                pred_pairs.pop()
                pred_events.pop()

        return train_pairs, train_events, pred_pairs, pred_events

#(self, nodes_num = 0, edges: torch.LongTensor = None, edge_times: torch.FloatTensor = None,  edge_states: torch.FloatTensor = None, directed:bool=None, verbose=False):

dataset = Dataset()
dataset.read_edgelist("./datasets/synthetic_april21_undirected/synthetic_april21_undirected.edges")

DataSplit(dataset=dataset)




# ########################################################################################################################

# # Set the seed value
# set_seed(seed=seed)

# # Create the target folder
# os.makedirs(output_folder)
# if log_file != "":
#     orig_stdout = sys.stdout
#     f = open(log_file, 'w')
#     sys.stdout = f

# # Read the dataset
# dataset = Dataset(path=input_folder, normalize=True, seed=seed)
# nodes_num = dataset.number_of_nodes()
# pairs, events = dataset.get_pairs(), dataset.get_events()

# ########################################################################################################################

# if verbose:
#     print("- The network is being divided into training and prediction sets for the future!")

# # Firstly, the network will be split into two part
# split_ratio = 1.0 - prediction_ratio

# train_pairs, train_events = [], []
# pred_pairs, pred_events = [], []
# for pair, pair_events in zip(pairs, events):

#     train_pairs.append(pair)
#     train_events.append([])

#     pred_pairs.append(pair)
#     pred_events.append([])

#     for e in pair_events:
#         if e <= split_ratio:
#             train_events[-1].append(e)
#         else:
#             pred_events[-1].append(e)

#     # Remove the pair if it does not contain any event
#     if len(train_events[-1]) == 0:
#         train_pairs.pop()
#         train_events.pop()

#     # Remove the pair if it does not contain any event
#     if len(pred_events[-1]) == 0:
#         pred_pairs.pop()
#         pred_events.pop()

# # Construct an undirected static graph from the links in the training set
# train_graph = nx.Graph()
# train_graph.add_edges_from(train_pairs)

# if verbose:
#     print(f"\t+ Training graph has {train_graph.number_of_nodes()} nodes.")
#     print(f"\t+ Training graph has {train_graph.number_of_edges()} pairs having at least one events.")
#     print(f"\t+ Prediction set has {len(np.unique(np.asarray(pred_pairs)))} nodes.")
#     print(f"\t+ Prediction set has {len(pred_pairs)} pairs having at least one events.")

# # If there are any nodes which do not have any events during the training timeline,
# # the graph must be relabeled and these nodes must be removed from the testing set as well.
# newlabel = None
# if len(list(nx.isolates(train_graph))) != 0:

#     isolated_nodes = list(nx.isolates(train_graph))
#     if verbose:
#         print(f"\t\t+ Training graph has {len(isolated_nodes)} isolated nodes.")

#     n, count = 0, 0
#     while n < len(pred_pairs):
#         i, j = pred_pairs[n]
#         if i in isolated_nodes or j in isolated_nodes:
#             pred_pairs.pop(n)
#             pred_events.pop(n)
#             count += 1
#         else:
#             n += 1

#     # Remove the isolated nodes from the networkx graph
#     train_graph.remove_nodes_from(isolated_nodes)

#     if verbose:
#         print(f"\t\t+ {count} pairs have been removed from the prediction set.")
#         print(f"\t\t+ The prediction set has currently {len(np.unique(np.asarray(pred_pairs)))} nodes.")
#         print(f"\t\t+ The prediction set has currently {len(pred_pairs)} pairs having at least one events.")

#     # Set the number of nodes
#     nodes_num = train_graph.number_of_nodes()

#     if verbose:
#         print(f"\t+ Nodes are being relabeled.")

#     # Relabel nodes in the training set
#     newlabel = {node: idx for idx, node in enumerate(train_graph.nodes())}
#     for n, pair in enumerate(train_pairs):
#         train_pairs[n] = [newlabel[pair[0]], newlabel[pair[1]]]

#     # Relabel nodes in the prediction set
#     for n, pair in enumerate(pred_pairs):
#         pred_pairs[n] = [newlabel[pair[0]], newlabel[pair[1]]]

#     # Relabel nodes in the networkx object
#     train_graph = nx.relabel_nodes(G=train_graph, mapping=newlabel)

#     if verbose:
#         print(f"\t\t+ Completed.")

# # Finally construct the train dataset. Event times must not be normalized.
# data_node2group = dataset.get_groups()
# if data_node2group is None:
#     node2group = None
# else:
#     node2group = {node if newlabel is None else newlabel[node]: data_node2group[node] for node in train_graph.nodes()}

# # newlabel = {node: idx for idx, node in enumerate(train_graph.nodes())}
# train_dataset = Dataset(
#     data=(train_events, train_pairs, list(range(nodes_num)), node2group),
#     normalize=False, seed=seed, verbose=False
# )

# ########################################################################################################################

# if verbose:
#     print("- Sampling processes for the masking and completion pairs have just started!")

# # Sample the masking and completion pairs
# all_possible_pair_num = nodes_num * (nodes_num - 1) // 2
# mask_size = int(all_possible_pair_num * masking_ratio)
# completion_size = int(all_possible_pair_num * completion_ratio)
# total_sample_size = mask_size + completion_size

# # Construct pair indices
# all_pos_pairs = [(i,j) for i in range(nodes_num) for j in range(i+1, nodes_num)] # all_pair_indices = list(range(all_possible_pair_num))
# np.random.shuffle(all_pos_pairs)
# # np.random.shuffle(all_pair_indices)

# # Sample node pairs such that each node in the residual has at least one event
# sampled_pairs = []
# for k, pair in enumerate(all_pos_pairs):
#     i, j = pair #linearIdx2matIdx(idx=pair_idx, n=nodes_num, k=2)

#     if train_graph.has_edge(i, j):
#         train_graph.remove_edge(i, j)

#         if len(list(nx.isolates(train_graph))) != 0:
#             train_graph.add_edge(i, j)
#         else:
#             sampled_pairs.append([i, j])
#     else:
#         sampled_pairs.append([i, j])

#     if len(sampled_pairs) == total_sample_size:
#         break

# assert len(sampled_pairs) == total_sample_size, "Enough number of sample pairs couldn't be found!"

# # Set the completion and mask pairs
# mask_pairs, completion_pairs = [], []
# if mask_size:
#     mask_pairs = sampled_pairs[:mask_size]
# if completion_size:
#     completion_pairs = sampled_pairs[mask_size:]

# # Set the completion and mask events
# mask_events = [train_dataset[pair][1] for pair in mask_pairs]
# completion_events = [train_dataset[pair][1] for pair in completion_pairs]

# # Construct the residual pairs and events
# # Since we always checked in the previous process, every node has at least one event
# residual_pairs, residual_events = train_pairs.copy(), train_events.copy()

# if completion_size:
#     completion_pair_indices = [pairIdx2flatIdx(pair[0], pair[1], nodes_num) for pair in completion_pairs]

#     n = 0
#     while n < len(residual_pairs):
#         pair = residual_pairs[n]
#         if pairIdx2flatIdx(pair[0], pair[1], nodes_num) in completion_pair_indices:
#             residual_pairs.pop(n)
#             residual_events.pop(n)
#         else:
#             n += 1

# print(len(np.unique(residual_pairs)))
# if verbose:
#     print(f"\t+ Masking set has {mask_size} pairs.")
#     mask_samples_event_pairs_num = sum([1 if len(pair_events) else 0 for pair_events in mask_events])
#     print(f"\t\t+ {mask_samples_event_pairs_num} masking pairs have at least one event. ")

#     print(f"\t+ Completion set has {completion_size} pairs.")
#     completion_samples_event_pairs_num = sum([1 if len(pair_events) else 0 for pair_events in completion_events])
#     print(f"\t\t+ {completion_samples_event_pairs_num} masking pairs have at least one event. ")

#     print(f"\t+ Residual network has {len(residual_pairs)} event pairs.")

# ########################################################################################################################

# if verbose:
#     print("- The files are being written...")

# # Save the training pair and events
# os.makedirs(os.path.join(output_folder, "train"))
# with open(os.path.join(output_folder, "train", "pairs.pkl"), 'wb') as f:
#     pickle.dump(train_pairs, f)
# with open(os.path.join(output_folder, "train", "events.pkl"), 'wb') as f:
#     pickle.dump(train_events, f)
# if node2group is not None:
#     with open(os.path.join(output_folder, "node2group.pkl"), 'wb') as f:
#         pickle.dump(dataset.get_groups(), f)

# # Save the residual pair and events
# os.makedirs(os.path.join(output_folder, "residual"))
# with open(os.path.join(output_folder, "residual", "pairs.pkl"), 'wb') as f:
#     pickle.dump(residual_pairs, f)
# with open(os.path.join(output_folder, "residual", "events.pkl"), 'wb') as f:
#     pickle.dump(residual_events, f)

# # Save the training pair and events
# os.makedirs(os.path.join(output_folder, "completion"))
# with open(os.path.join(output_folder, "completion", "pairs.pkl"), 'wb') as f:
#     pickle.dump(completion_pairs, f)
# with open(os.path.join(output_folder, "completion", "events.pkl"), 'wb') as f:
#     pickle.dump(completion_events, f)

# # Save the mask pairs
# os.makedirs(os.path.join(output_folder, "mask"))
# with open(os.path.join(output_folder, "mask", "pairs.pkl"), 'wb') as f:
#     pickle.dump(mask_pairs, f)
# with open(os.path.join(output_folder, "mask", "events.pkl"), 'wb') as f:
#     pickle.dump(mask_events, f)

# # Save the prediction pairs
# os.makedirs(os.path.join(output_folder, "prediction"))
# with open(os.path.join(output_folder, "prediction", "pairs.pkl"), 'wb') as f:
#     pickle.dump(pred_pairs, f)
# with open(os.path.join(output_folder, "prediction", "events.pkl"), 'wb') as f:
#     pickle.dump(pred_events, f)

# if verbose:
#     print(f"\t+ Completed.")

# ########################################################################################################################

# if log_file != "":
#     sys.stdout = orig_stdout
#     f.close()