import re
import math

class Dataset:

    def __init__(self, edge_file: str = None) -> None:
        
        self.__edges, self.__nodes_num, self.__min_value, self.__max_value = self.read_edgelist(edge_file)

    def read_edgelist(self, file_path):
        '''
        :param file_path: path of the file 
        '''
        
        min_value, max_value = math.inf, -math.inf
        edges, nodes = [], []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                # Discard the lines starting with #
                if line[0] != '#':
                    
                    # Split the line into a list of strings
                    tokens = re.split(';|,| |\n', line.strip())
                    edges.append( tuple( int(token) for token in tokens[:-1] ) )

                    try:
                        edges[-1] = edges[-1] + (int(tokens[-1]), )
                    except ValueError:
                        edges[-1] = edges[-1] + (float(tokens[-1]), )

                    # Add the nodes
                    nodes.append(int(tokens[0]))
                    nodes.append(int(tokens[1]))

                    # Get the min and max values
                    min_value = min(min_value, edges[-1][-1])
                    max_value = max(max_value, edges[-1][-1])
        
        nodes = list(set(nodes))
        assert min(nodes) == 0, f"The nodes must be numbered from 0 to N-1, min node: {min(nodes)}."
        assert max(nodes) + 1 == len(nodes), f"The nodes must be numbered from 0 to N-1, max node: {max(nodes)}."

        # Set the number of nodes
        nodes_num = len(nodes)

        return edges, nodes_num, min_value, max_value

    def get_edges(self):
        return self.__edges
    
    def get_nodes_num(self):
        return self.__nodes_num

    def get_min_value(self):
        return self.__min_value

    def get_max_value(self):
        return self.__max_value

# class Dataset:

#     def __init__(self, path: str = None, data: tuple = None, normalize: bool = True,
#                  verbose: bool = True, seed: int = 19):

#         self.__events = None
#         self.__pairs = None
#         self.__node2group = None
#         self.__nodes_num = 0
#         self.__nodes = []
#         self.__verbose = verbose

#         assert path is None or data is None, "Path and data parameter cannot be set at the same time!"

#         if path is not None:
#             self.read(path, normalize=normalize)

#         if data is not None:
#             self.__events = data[0]
#             self.__pairs = data[1]

#             self.__nodes = data[2]
#             self.__nodes_num = len(self.__nodes)

#             if len(data) > 3:
#                 self.__node2group = data[3]

#             if self.__verbose:
#                 self.print_statistics()

#             if normalize:
#                 self.__normalize_events()

#         # Set the seed value
#         set_seed(seed=seed)

#     def read(self, path, normalize=True):

#         with open(os.path.join(path, 'events.pkl'), 'rb') as f:
#             self.__events = list(pkl.load(f))

#         with open(os.path.join(path, 'pairs.pkl'), 'rb') as f:
#             self.__pairs = np.asarray(pkl.load(f), dtype=int).tolist()

#         self.__nodes = np.unique(self.__pairs).tolist()
#         self.__nodes_num = len(self.__nodes)

#         if self.__verbose:
#             self.print_statistics()

#         if normalize:
#             self.__normalize_events()

#         # Read the file storing the groups of nodes if exists
#         node2group_file_path = os.path.join(path, 'node2group.pkl')
#         if os.path.exists(node2group_file_path):
#             with open(node2group_file_path, 'rb') as f:
#                 self.__node2group = pkl.load(f)

#     def __normalize_events(self):

#         min_time = self.get_min_event_time()
#         max_time = self.get_max_event_time()

#         if min_time != 0.0 or max_time != 1.0:

#             if self.__verbose:
#                 print(f"- The event times are being normalized...")
#                 print(f"\t+ The minimum time: {min_time}")
#                 print(f"\t+ The maximum time: {max_time}")

#             for idx, e in enumerate(self.__events):
#                 self.__events[idx] = (e - min_time ) / float(max_time - min_time)

#     def write(self, folder_path):

#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         with open(os.path.join(folder_path, 'events.pkl'), 'wb') as f:
#             pkl.dump(self.__events, f)

#         with open(os.path.join(folder_path, 'pairs.pkl'), 'wb') as f:
#             pkl.dump(self.__pairs, f)

#     def __getitem__(self, item):

#         if type(item) is int:

#             return self.__pairs[item], self.__events[item]

#         elif type(item) is tuple or type(item) is list:

#             try:
#                 idx = self.__pairs.index(list(item))
#                 return self.__pairs[idx], self.__events[idx]

#             except ValueError:
#                 return item, []

#         else:

#             raise ValueError("Invalid input type!")

#     def number_of_nodes(self):

#         return self.__nodes_num

#     def number_of_pairs(self):

#         return len(self.__events)

#     def number_of_events(self):

#         return len(self.__events)

#     def get_min_event_time(self):

#         return min(self.__events)

#     def get_max_event_time(self):

#         return max(self.__events)

#     def get_nodes(self):

#         return self.__nodes

#     def get_events(self):

#         return self.__events

#     def get_pairs(self):

#         return self.__pairs

#     def get_groups(self):

#         return self.__node2group

#     def print_statistics(self):

#         print(f"- The dataset statistics:")
#         print(f"\t+ The number of nodes: {self.number_of_nodes()}")
#         print(f"\t+ The total number of edges: {self.number_of_events()}")
#         print(f"\t+ The initial time of the dataset: {self.get_min_event_time()}")
#         print(f"\t+ The last time of the dataset: {self.get_max_event_time()}")

#     def plot_events(self, nodes: list = None, fig_size: tuple = None, show = True):

#         if nodes is None:
#             nodes = [0, 1]

#         nodes_num = len(nodes)
#         nodes = sorted(nodes)

#         pair_indices = [[i, j] for i in range(nodes_num) for j in range(i + 1, nodes_num)]
#         pairs = [[nodes[i], nodes[j]] for i, j in pair_indices]

#         plt.figure(figsize=fig_size if fig_size is not None else (12, 10))

#         for pairIdx, pair in enumerate(pairs):
#             _, events = self.__getitem__(pair)
#             y = len(events) * [pairIdx]
#             plt.plot(events, y, 'k.')

#         plt.yticks(np.arange(len(pairs)), [f"({pair[0]},{pair[1]})" for pair in pairs])

#         plt.xlabel("Timeline")
#         plt.ylabel("Node pairs")

#         if show:
#             plt.show()

#         return plt

#     def plot_samples(self, labels, samples, fig_size: tuple = None):

#         # Plot the events
#         plt = self.plot_events(nodes=list(range(self.number_of_nodes())), fig_size= fig_size, show=False)

#         # Check if the samples contain event times
#         assert len(samples[0]) == 3, "Samples do not contain event times!"

#         c = ['r.', 'b.']
#         for label, sample in zip(labels, samples):

#             plt.plot(sample[2], pairIdx2flatIdx(i=sample[0], j=sample[1], n=self.number_of_nodes()), c[label])

#         plt.show()

#     def get_freq(self):

#         F = np.zeros(shape=(self.__nodes_num, self.__nodes_num), dtype=np.int)

#         for i, j in zip(*np.triu_indices(self.__nodes_num, k=1)):
#             F[i, j] = len(self[[i, j]][1])

#         return F

#     def info(self):

#         print("- Dataset Information -")
#         print(f"\tNumber of nodes: {self.number_of_nodes()}")
#         print(f"\tNumber of events: {self.number_of_total_events()}")
#         p = round(100 * self.number_of_event_pairs()/(0.5 * self.number_of_nodes() * (self.number_of_nodes() - 1)), 2)
#         print(f"\tNumber of pairs having at least one event: {self.number_of_event_pairs()} ({p}%)")
#         print(f"\tAverage number of events per pair: {self.number_of_total_events() / float(len(self.__pairs))}")
#         print(f"\tMin. time: {self.get_min_event_time()}")
#         print(f"\tMax. time: {self.get_max_event_time()}")