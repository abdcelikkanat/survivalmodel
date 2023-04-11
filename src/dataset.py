import re
from utils.common import INF
import torch

class Dataset:

    def __init__(self, nodes_num = 0, edges: torch.LongTensor = None, edge_times: torch.FloatTensor = None, 
                edge_states: torch.FloatTensor = None, init_time:float = None, last_time:float = None, verbose=False):

        self.__nodes_num = nodes_num
        self.__edges = edges
        self.__edge_times = edge_times
        self.__edge_states = edge_states
        self.__init_time = init_time
        self.__last_time = last_time
        self.__verbose = verbose
        
    def read_edgelist(self, file_path):
        '''
        Read the edge list file
        :param file_path: path of the file 
        '''
        
        min_value, max_value = INF, -INF
        data = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                # Discard the lines starting with #
                if line[0] == '#':
                    pass

                # Lines starting with > provide the information about the initial and last time
                elif line[0] == '>':
            
                    tokens = tuple( float(value) for value in line[1:].strip().split() )
                    self.__init_time, self.__last_time = tokens[0], tokens[1]

                else:
                    
                    # Split the line into a list of strings
                    tokens = re.split(';|,| |\n', line.strip())
                    # Convert the values to float values
                    current_line = tuple( float(token) for token in tokens )

                    # Add the edge
                    data.append(current_line)

        # Construct a tensor from the list
        data = torch.as_tensor(data, dtype=torch.float)

        # Get the shape of the data
        edges_num, data_dim = data.shape

        # Split the columns
        self.__edges = data[:, :2].to(torch.long)
        self.__edge_times = data[:, 2]
        self.__edge_states = data[:, 3] if data_dim > 3 else None

        # Get the nodes
        nodes = torch.unique(self.__edges)

        # Check the minimum and maximum node labels
        assert min(nodes) == 0, f"The nodes must be numbered from 0 to N-1, min node: {min(nodes)}/{len(nodes)}."
        assert max(nodes) + 1 == len(nodes), f"The nodes must be numbered from 0 to N-1, max node: {max(nodes)}/{len(nodes)}."
        
        # Construct a node list
        nodes = torch.unique(self.__edges)
        self.__nodes_num = len(nodes)

        # Sort the edges
        sorted_indices = self.__edge_times.argsort()
        self.__edges = self.__edges[sorted_indices, :]
        self.__edge_times = self.__edge_times[sorted_indices]
        self.__edge_states = None if self.__edge_states is None else self.__edge_states[sorted_indices]

        # If the minimum and maximum time are not given, set them
        if self.__init_time is None:
            self.__init_time = self.__edge_times.min()
        if self.__last_time is None:
            self.__last_time = self.__edge_times.max()

        if self.__verbose:
            print(f"Number of nodes: {self.__nodes_num}")
            print(f"Number of edges: {edges_num}")
            print(f"Initial time: {self.__init_time}")
            print(f"Last time: {self.__last_time}")

    def get_edges(self) -> torch.Tensor:
        '''
        Get the edges
        '''
        if self.__edge_states is None:
            return zip(map(tuple, self.__edges.tolist()), self.__edge_times.tolist())
        else:
            return zip(map(tuple, self.__edges.tolist()), self.__edge_times.tolist(), self.__edge_states.tolist())
    
    def get_nodes_num(self) -> int:
        '''
        Get the number of nodes
        '''
        return self.__nodes_num

    def split_over_time(self, split_ratio: float = 0.9, remove_isolated_nodes: bool = True):
        '''
        Split the edges over time
        :param split_ratio: ratio of the edges in the first split
        :param keep_isolated_nodes: if True, the isolated nodes are kept in the first split (if exists)
        '''

        # Get the min and max time
        split_time = (self.__last_time - self.__init_time) * split_ratio + self.__init_time
        if self.__verbose:
            print(f"Split time: {split_time}")

        # Get the indices of the edges with time less than the split time
        split_index = torch.searchsorted(self.__edge_times, split_time)

        # Split the edges
        # First split
        edges1 = self.__edges[:split_index, :]
        edge_times1 = self.__edge_times[:split_index]
        edge_states1 = None if self.__edge_states is None else self.__edge_states[:split_index] 
        
        init_time1 = self.__init_time
        last_time1 = split_time

        # Second split
        edges2 = self.__edges[split_index:, :]
        edge_times2 = self.__edge_times[split_index:]
        edge_states2 = None if self.__edge_states is None else self.__edge_states[split_index:] 

        init_time2 = split_time
        last_time2 = self.__last_time

        nodes_num = self.__nodes_num

        # Remove the isolated nodes in the (undirected version of) the graph
        if remove_isolated_nodes:

            # Get the isolated nodes in the first split
            nonisolated_nodes = torch.sort(torch.unique(edges1))[0]

            # Check if there are isolated nodes in the first split, if so, relabel the edges
            if len(nonisolated_nodes) < self.__nodes_num:
                new_labels = torch.zeros(self.__nodes_num, dtype=torch.long)
                new_labels[nonisolated_nodes] = torch.arange(len(nonisolated_nodes), dtype=torch.long)

                # Relabel the edges in the first split
                ### Note that there is no need to remove any edges from the first split
                edges1[:, 0], edges1[:, 1] = new_labels[edges1[:, 0]], new_labels[edges1[:, 1]]

                # Remove the edges in the second split, containing isolated nodes
                mask = (edges2[:, 0].unsqueeze(1) == nonisolated_nodes).any(dim=1) & (edges2[:, 1].unsqueeze(1) == nonisolated_nodes).any(dim=1)
                edges2 = edges2[mask, :]
                edge_times2 = edge_times2[mask]
                edge_states2 = None if edge_states2 is None else edge_states2[mask]

                # Relabel the edges in the second split
                edges2[:, 0], edges2[:, 1] = new_labels[edges2[:, 0]], new_labels[edges2[:, 1]]

                nodes_num = len(nonisolated_nodes)


        ds1 = Dataset(
            nodes_num = nodes_num, edges=edges1, edge_times=edge_times1, 
            edge_states=edge_states1, init_time=init_time1, last_time=last_time1
        )
        ds2 = Dataset(
            nodes_num = nodes_num, edges=edges2, edge_times=edge_times2, 
            edge_states=edge_states2, init_time=init_time2, last_time=last_time2
        )

        return ds1, ds2


    def split(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):

        return None