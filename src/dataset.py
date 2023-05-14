import re

import utils
from utils.common import INF
import torch

class Dataset:

    def __init__(self, nodes_num = 0, edges: torch.LongTensor = None, edge_times: torch.Tensor = None,
                edge_states: torch.Tensor = None, directed: bool = None, signed: bool = None, verbose=False):

        self.__nodes_num = nodes_num
        self.__edges = edges
        self.__times = edge_times
        self.__states = edge_states
        self.__directed = directed
        self.__signed = signed
        self.__verbose = verbose

        # If the edge times are given, find the initial and last time
        if self.__times is not None:
            self.__init_time = self.__times.min() 
            self.__last_time = self.__times.max()

        # Check if the given parameters are valid
        if self.__edges is not None:
            assert self.__edges.shape[0] == 2, \
                "The edges must be a matrix of shape (2xT)!"
            assert self.__edges.shape[1] == self.__times.shape[0], \
                f"The number of edges ({self.__edges.shape[1]}) and the length of edge times ({self.__times.shape[0]}) must match!"
        
    def read_edgelist(self, file_path):
        '''
        Read the edge list file
        :param file_path: path of the file 
        '''
        
        edges, edge_times, edge_states = [], [], []
        self.__directed, self.__signed = False, False
        with open(file_path, 'r') as f:
            for line in f.readlines():
                # Discard the lines starting with #
                if line[0] == '#':
                    pass

                # Lines starting with > provide the information about the initial and last time
                elif line[:5] == '!INFO':
                    
                    pass
                    # tokens = tuple( float(value) for value in line[1:].strip().split()

                else:
                    
                    # Split the line into a list of strings
                    tokens = re.split(';|,| |\t|\n', line.strip())

                    if len(tokens) < 3:
                        raise Exception("An edge must consist of at least 3 columns: source, target and time!")

                    # Add the edge
                    edges.append( (int(tokens[0]), int(tokens[1])) )

                    # If the first node of the edge is greater than the second, the graph is directed
                    if edges[-1][0] > edges[-1][1]:
                        self.__directed = True
                    
                    if edges[-1][0] == edges[-1][1]:
                        raise ValueError("Self loops are not allowed!")

                    # Add the time
                    edge_times.append( int(tokens[2]) )

                    # Add the state if given
                    if len(tokens) > 3:
                        edge_states.append( int(tokens[3]) )

                        if int(tokens[3]) < 0:
                            self.__signed = True

        # Split the columns
        self.__edges = torch.as_tensor(edges, dtype=torch.long).T
        self.__times = torch.as_tensor(edge_times, dtype=torch.long) # Unix timestamp
        self.__states = torch.as_tensor(edge_states, dtype=torch.long) if len(edge_states) > 0 else None

        # Get the nodes
        nodes = torch.unique(self.__edges)

        # Check the minimum and maximum node labels
        assert min(nodes) == 0, f"The nodes must be numbered from 0 to N-1, min node: {min(nodes)}/{len(nodes)}."
        assert max(nodes) + 1 == len(nodes), f"The nodes must be numbered from 0 to N-1, max node: {max(nodes)}/{len(nodes)}."
        
        # Construct a node list
        nodes = torch.unique(self.__edges)
        self.__nodes_num = len(nodes)

        # Sort the edges
        sorted_indices = self.__times.argsort()
        self.__edges = self.__edges[:, sorted_indices]
        self.__times = self.__times[sorted_indices]
        self.__states = self.__states[sorted_indices] if len(edge_states) > 0 else None

        # If the minimum and maximum time are not given, set them
        self.__init_time = self.__times.min()
        self.__last_time = self.__times.max()

        if self.__verbose:
            self.print_info()
    
    def get_nodes_num(self) -> int:
        '''
        Get the number of nodes
        '''
        return self.__nodes_num

    def get_init_time(self) -> float:
        '''
        Get the iniitial time
        '''
        return self.__init_time

    def set_init_time(self, init_time: float):
        '''
        Set the initial time
        '''
        self.__init_time = init_time

    def get_last_time(self) -> float:
        '''
        Get the last time
        '''
        return self.__last_time

    def set_last_time(self, last_time: float):
        '''
        Set the last time
        '''
        self.__last_time = last_time

    def write_data(self, file_path: str):

        # Write the edges
        with open(file_path, 'w') as f:
            for edge, time, state in zip(self.get_edges().T, self.get_times(), self.get_states()):
                f.write(f"{edge[0]}\t{edge[1]}\t{time}\t{state}\n")

    # def split_over_time(self, split_ratio: float = 0.9, remove_isolated_nodes: bool = True):
    #     '''
    #     Split the edges over time
    #     :param split_ratio: ratio of the edges in the first split
    #     :param keep_isolated_nodes: if True, the isolated nodes are kept in the first split (if exists)
    #     '''
    # 
    #     # Get the min and max time
    #     split_time = (self.__last_time - self.__init_time) * split_ratio + self.__init_time
    #     if self.__verbose:
    #         print(f"Split time: {split_time}")
    # 
    #     # Get the indices of the edges with time less than the split time
    #     split_index = torch.searchsorted(self.__times, split_time)
    # 
    #     # Split the edges
    #     # First split
    #     edges1 = self.__edges[:, :split_index]
    #     edge_times1 = self.__times[:split_index]
    #     edge_states1 = None if self.__states is None else self.__states[:split_index] 
    #     
    #     init_time1 = self.__init_time
    #     last_time1 = split_time
    # 
    #     # Second split
    #     edges2 = self.__edges[:, split_index:]
    #     edge_times2 = self.__times[split_index:]
    #     edge_states2 = None if self.__states is None else self.__states[split_index:] 
    # 
    #     init_time2 = split_time
    #     last_time2 = self.__last_time
    # 
    #     nodes_num = self.__nodes_num
    # 
    #     # Remove the isolated nodes in the (undirected version of) the graph
    #     if remove_isolated_nodes:
    # 
    #         # Get the isolated nodes in the first split
    #         nonisolated_nodes = torch.sort(torch.unique(edges1))[0]
    # 
    #         # Check if there are isolated nodes in the first split, if so, relabel the edges
    #         if len(nonisolated_nodes) < self.__nodes_num:
    #             new_labels = torch.zeros(self.__nodes_num, dtype=torch.long)
    #             new_labels[nonisolated_nodes] = torch.arange(len(nonisolated_nodes), dtype=torch.long)
    # 
    #             # Relabel the edges in the first split
    #             ### Note that there is no need to remove any edges from the first split
    #             edges1[:, 0], edges1[:, 1] = new_labels[edges1[:, 0]], new_labels[edges1[:, 1]]
    # 
    #             # Remove the edges in the second split, containing isolated nodes
    #             mask = (edges2[:, 0].unsqueeze(1) == nonisolated_nodes).any(dim=1) & (edges2[:, 1].unsqueeze(1) == nonisolated_nodes).any(dim=1)
    #             edges2 = edges2[mask, :]
    #             edge_times2 = edge_times2[mask]
    #             edge_states2 = None if edge_states2 is None else edge_states2[mask]
    # 
    #             # Relabel the edges in the second split
    #             edges2[:, 0], edges2[:, 1] = new_labels[edges2[:, 0]], new_labels[edges2[:, 1]]
    # 
    #             nodes_num = len(nonisolated_nodes)
    # 
    # 
    #     ds1 = Dataset(
    #         nodes_num = nodes_num, edges=edges1, edge_times=edge_times1, 
    #         edge_states=edge_states1, init_time=init_time1, last_time=last_time1
    #     )
    #     ds2 = Dataset(
    #         nodes_num = nodes_num, edges=edges2, edge_times=edge_times2, 
    #         edge_states=edge_states2, init_time=init_time2, last_time=last_time2
    #     )
    # 
    #     return ds1, ds2
    # 
    # 
    # def split(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    # 
    #     return None

    def get_data(self, states=False):
        """"
        Get all data
        :param states: if True, return the states
        :return: edges, times, states (if states=True)
        """

        if states:
            return self.get_edges(), self.get_times(), self.get_states()
        else:
            return self.get_edges(), self.get_times()

    def get_edges(self, idx: int = None) -> torch.Tensor:
        """
        Get the edges
        """
        if idx is None:
            return self.__edges
        else:
            return self.__edges[idx]

    def get_times(self) -> torch.Tensor:
        """
        Get the edge times
        """

        return self.__times

    def get_states(self) -> torch.Tensor:
        """
        Get the states
        NOTE: It is assumed that the network is binary, not signed
        """

        # If states is None, construct the states tensor
        if self.__states is None:
            # Represent the edges in a flat index format
            edge_flat_idx = utils.matIdx2flatIdx(self.get_edges(0), self.get_edges(1), self.__nodes_num, self.__directed)
            # Get the unique indices and their counts
            _, idx_counts = torch.unique(edge_flat_idx, sorted=True, return_counts=True)
            # Construct a mask
            mask = torch.arange(max(idx_counts)).expand(len(idx_counts), -1) < idx_counts.unsqueeze(1)
            # Construct a tensor consisting of consecutive 0s and 1s
            zero_one_tensor = torch.zeros(max(idx_counts), dtype=torch.long)
            zero_one_tensor[torch.arange(max(idx_counts)) % 2 == 0] = 1
            # Construct the states tensor which is in the ordered  format
            ordered_states = zero_one_tensor.expand(len(idx_counts), -1)[mask]
            # Construct the states in the correct order with respect to the edges
            idx = torch.argsort(edge_flat_idx, stable=True).argsort(stable=True)

            return ordered_states[idx]

        else:

            return self.__states

    def get_data_dict(self):
        """
        Get the data in a dictionary format
        :return: a dictionary with keys the source nodes and values a dictionary with keys the target nodes and values
        """

        # adj_list = [[] for _ in range(self.__nodes_num)]
        #
        # for i, j, t, s in zip(self.get_edges(0), self.get_edges(1), self.get_times(), self.get_states()):
        #     adj_list[i].append((j, t, s))
        #
        # return adj_list

        data_dict = {}
        for i, j, t, s in zip(self.get_edges(0), self.get_edges(1), self.get_times(), self.get_states()):

            if i.item() in data_dict:
                if j.item() in data_dict[i.item()]:
                    data_dict[i.item()][j.item()].append((t, s))
                else:
                    data_dict[i.item()][j.item()] = [(t, s)]
            else:
                data_dict[i.item()] = {j.item(): [(t, s)]}

        return data_dict

    def has_isolated_nodes(self):
        '''
        Check if the graph has isolated nodes
        '''

        # Get the isolated nodes in the first split
        nonisolated_nodes = torch.sort(torch.unique(self.__edges))[0]

        return self.__nodes_num - len(nonisolated_nodes)

    def is_directed(self):
        """
        Check if the graph is directed
        """

        return self.__directed

    def is_signed(self):
        """
        Check if the graph is signed
        """

        return self.__signed

    def print_info(self):
        """
        Print the dataset info
        """
        print(f"+ Dataset information")
        print(f"\t- Number of nodes: {self.__nodes_num}")
        print(f"\t- Number of edges: {self.__edges.shape[1]}")
        print(f"\t- Is directed: {self.__directed}")
        print(f"\t- Is signed: {self.__signed}")
        print(f"\t- Initial time: {self.__init_time}")
        print(f"\t- Last time: {self.__last_time}")