import torch
import utils

class BatchSampler(torch.nn.Module):
    
    def __init__(self, edges: torch.LongTensor, edge_times: torch.FloatTensor, edge_states: torch.LongTensor, 
                bin_bounds: torch.FloatTensor, nodes_num: int, batch_size: int, directed: bool, device: torch.device = "cpu", seed: int = 19):

        super(BatchSampler, self).__init__()

        # Set the parameters
        self.__edges = edges
        self.__edge_times = edge_times
        self.__edge_states = edge_states
        self.__nodes_num = nodes_num
        self.__batch_size = batch_size if batch_size > 0 else nodes_num
        self.__directed = directed
        self.__device = device

        # Define the possible number of pairs
        self.__pairs_num = self.__nodes_num * (self.__nodes_num - 1) if self.__directed else self.__nodes_num * (self.__nodes_num - 1) // 2

        # Convert the edges to flat indices
        self.__edges_flat_idx = utils.pairIdx2flatIdx(self.__edges[0], self.__edges[1], n=self.__nodes_num, directed=self.__directed)
        _, counts = torch.unique(self.__edges_flat_idx, sorted=True, return_inverse=False, return_counts=True)
        max_edge_count_per_pair = max(counts)
        indices = torch.argsort(self.__edges_flat_idx, stable=True)
        range_vector = torch.arange(max_edge_count_per_pair).expand(counts.shape[0], max_edge_count_per_pair)
        mask = (range_vector < counts.unsqueeze(1))
        edge_mat_indices = range_vector[mask][indices.argsort(stable=True)]

        # Construct a matrix of shape (pairs_num x max_edge_count_per_pair) for edge times
        self.__edge_times_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((self.__edges_flat_idx, edge_mat_indices)), 
            values=self.__edge_times,
            size=(self.__pairs_num, max_edge_count_per_pair), device=self.__device
        )

        # Construct the sparse state matrix
        ### It must be of float type due to the torch issues
        self.__edge_states_mat =  torch.sparse_coo_tensor(
            indices=torch.vstack((self.__edges_flat_idx, edge_mat_indices)), 
            values=self.__edge_states.to(torch.float),
            size=(self.__pairs_num, max_edge_count_per_pair), device=self.__device
        )
        utils.set_seed(seed)

    def sample(self):

        # Sample the batch nodes
        batch_nodes = torch.multinomial(torch.arange(self.__nodes_num, dtype=torch.float, device=self.__device), self.__batch_size, replacement=False).to(torch.long)
        # Sort the nodes in order to obtain pairs (i,j) such that i < j otherwise there might exist j > i pairs violating undirected case
        batch_nodes = batch_nodes.sort()[0]
        # Construct a matrix of shape 2x(Batch Size) storing the all possible batch node pairs
        batch_pairs = torch.combinations(batch_nodes, with_replacement=False).T
        if self.__directed:
            batch_pairs = torch.hstack((batch_pairs, torch.flip(batch_pairs, dims=(0, ))))
        # Convert the batch pairs in flat indices
        batch_flat_idx = utils.pairIdx2flatIdx(batch_pairs[0], batch_pairs[1], n=self.__nodes_num, directed=self.__directed)

        # Construct a diagonal matrix of shape (pairs_num x pairs_num) to select batch edges
        selection_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((batch_flat_idx, batch_flat_idx)), 
            values=torch.ones(len(batch_flat_idx), dtype=torch.float, device=self.__device), 
            size=(self.__pairs_num, self.__pairs_num), device=self.__device
        )

        output = torch.sparse.mm(selection_mat, self.__edge_times_mat)
        # Construct the batch edges
        batch_edges = utils.linearIdx2matIdx(output.indices()[0], n=self.__nodes_num, directed=self.__directed)
        # Construct the batch edge times
        batch_edge_times = output.values()
        # Construct the batch edge states
        output = torch.sparse.mm(selection_mat, self.__edge_states_mat)
        batch_states = output.values()

        return batch_nodes, batch_pairs, batch_edges, batch_edge_times, batch_states.to(torch.long)

