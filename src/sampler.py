import torch
import utils


class BatchSampler(torch.nn.Module):
    
    def __init__(self, edges: torch.LongTensor, edge_times: torch.FloatTensor, edge_states: torch.LongTensor,
                 nodes_num: int, batch_size: int, directed: bool, bin_bounds: torch.FloatTensor,
                 device: torch.device = "cpu", seed: int = 19):

        super(BatchSampler, self).__init__()

        # Set the parameters
        self.__bin_bounds = bin_bounds
        self.__edges = edges
        self.__edge_times = edge_times
        self.__edge_states = edge_states
        self.__nodes_num = nodes_num
        self.__batch_size = batch_size if batch_size > 0 else nodes_num
        self.__directed = directed
        self.__device = device

        # Define the possible number of pairs
        self.__pairs_num = self.__nodes_num * (self.__nodes_num - 1)
        if not self.__directed:
            self.__pairs_num = self.__pairs_num // 2

        # Convert the edges to flat indices
        self.__edges_flat_idx = utils.matIdx2flatIdx(
            self.__edges[0], self.__edges[1], n=self.__nodes_num, is_directed=self.__directed
        )
        _, counts = torch.unique(self.__edges_flat_idx, sorted=True, return_inverse=False, return_counts=True)
        max_edge_count_per_pair = max(counts)
        indices = torch.argsort(self.__edges_flat_idx, stable=True)
        range_vector = torch.arange(max_edge_count_per_pair, device=self.__device).expand(counts.shape[0], max_edge_count_per_pair)
        mask = (range_vector < counts.unsqueeze(1))
        edge_mat_indices = range_vector[mask][indices.argsort(stable=True)]

        # Construct a matrix of shape (pairs_num x max_edge_count_per_pair) for edge times
        self.__edge_times_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((self.__edges_flat_idx, edge_mat_indices)), 
            values=self.__edge_times,
            size=(self.__pairs_num, max_edge_count_per_pair), device=self.__device
        )

        # Construct the sparse state matrix
        # - It must be of float type due to the torch issues
        self.__edge_states_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((self.__edges_flat_idx, edge_mat_indices)), 
            values=self.__edge_states.to(torch.float),
            size=(self.__pairs_num, max_edge_count_per_pair), device=self.__device
        )
        utils.set_seed(seed)

    def sample(self):
        """
        Sample a batch of nodes and edges among them
        """
        # Sample the batch nodes
        batch_nodes = torch.multinomial(
            torch.arange(self.__nodes_num, dtype=torch.float, device=self.__device),
            self.__batch_size, replacement=False
        ).to(torch.long)

        # Sort the nodes in order to obtain pairs (i,j) such that i < j
        # Otherwise there might exist j > i pairs violating undirected case
        batch_nodes = batch_nodes.sort()[0]
        # Construct a matrix of shape 2x(Batch Size) storing the all possible batch node pairs
        batch_pair_combin = torch.combinations(batch_nodes, with_replacement=False).T
        if self.__directed:
            batch_pair_combin = torch.hstack((batch_pair_combin, torch.flip(batch_pair_combin, dims=(0,))))
        # Convert the batch pairs in flat indices
        batch_flat_idx_combin = utils.matIdx2flatIdx(
            batch_pair_combin[0], batch_pair_combin[1], self.__nodes_num, self.__directed
        )

        # Construct a diagonal matrix of shape (pairs_num x pairs_num) to select batch edges
        selection_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((batch_flat_idx_combin, batch_flat_idx_combin)),
            values=torch.ones(len(batch_flat_idx_combin), dtype=torch.float, device=self.__device),
            size=(self.__pairs_num, self.__pairs_num), device=self.__device
        )

        print( self.__edge_times_mat.to_dense())
        raise Exception("stop")

        output = torch.sparse.mm(selection_mat, self.__edge_times_mat)
        # Construct the batch edge times
        batch_times = output.values()
        # Construct the batch edge states
        batch_states = torch.sparse.mm(selection_mat, self.__edge_states_mat).values()

        expanded_pairs, expanded_times, expanded_states, event_states, is_edge, delta_t = utils.expand_data(
            nodes_num=self.__nodes_num, directed=self.__directed, bin_bounds=self.__bin_bounds,
            edge_pair_flat_idx=output.indices()[0], edge_times=batch_times, edge_states=batch_states,
            border_pair_flat_idx=batch_flat_idx_combin, device=self.__device
        )
        print(batch_nodes)
        return batch_nodes, expanded_pairs, expanded_times, expanded_states, event_states, is_edge, delta_t, \
            utils.flatIdx2matIdx(output.indices()[0], self.__nodes_num, self.__directed), batch_times, batch_states


