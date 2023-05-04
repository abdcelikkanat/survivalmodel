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
        self.__edges_flat_idx = utils.pairIdx2flatIdx(
            self.__edges[0], self.__edges[1], n=self.__nodes_num, directed=self.__directed
        )
        _, counts = torch.unique(self.__edges_flat_idx, sorted=True, return_inverse=False, return_counts=True)
        max_edge_count_per_pair = max(counts)
        indices = torch.argsort(self.__edges_flat_idx, stable=True)
        range_vector = torch.arange(max_edge_count_per_pair).expand(counts.shape[0], max_edge_count_per_pair)
        mask = (range_vector < counts.unsqueeze(1))
        edge_mat_indices = range_vector[mask][indices.argsort(stable=True)]

        # Construct a matrix of shape (pairs_num x max_edge_count_per_pair) for edge times
        self.__edge_times_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((self.__edges_flat_idx, edge_mat_indices)), values=self.__edge_times,
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

    def sample(self, bin_idx: int = None):
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
        batch_flat_idx_combin = utils.pairIdx2flatIdx(
            batch_pair_combin[0], batch_pair_combin[1], self.__nodes_num, self.__directed
        )

        # Construct a diagonal matrix of shape (pairs_num x pairs_num) to select batch edges
        selection_mat = torch.sparse_coo_tensor(
            indices=torch.vstack((batch_flat_idx_combin, batch_flat_idx_combin)),
            values=torch.ones(len(batch_flat_idx_combin), dtype=torch.float, device=self.__device),
            size=(self.__pairs_num, self.__pairs_num), device=self.__device
        )

        output = torch.sparse.mm(selection_mat, self.__edge_times_mat)
        # Construct the batch edges
        batch_edges = utils.linearIdx2matIdx(output.indices()[0], n=self.__nodes_num, directed=self.__directed)
        # Construct the batch edge times
        batch_times = output.values()
        # Construct the batch edge states
        batch_states = torch.sparse.mm(selection_mat, self.__edge_states_mat).values()

        bins_num = len(self.__bin_bounds) - 1
        # Expand the batch edges, times and states with the bin border times
        expanded_pair_idx = torch.concat((
            batch_flat_idx_combin.repeat_interleave(repeats=bins_num, dim=0),
            output.indices()[0]
        ))
        expanded_times = torch.concat((
            self.__bin_bounds[:-1].repeat(len(batch_flat_idx_combin)),
            batch_times
        ))
        expanded_states = torch.concat((
            torch.zeros(bins_num * len(batch_flat_idx_combin), dtype=torch.float, device=self.__device),
            batch_states.to(torch.float)
        ))
        # Mark the first time of bin borders to detect later the pair edges
        border_marked_idx = torch.concat((
            torch.eye(1, bins_num, dtype=torch.float, device=self.__device).squeeze(0).repeat(len(batch_flat_idx_combin)),
            torch.ones(len(batch_states), dtype=torch.float, device=self.__device)
        ))
        # An indicator vector to detect additional edge times
        is_edge = torch.concat((
            torch.zeros(bins_num * len(batch_flat_idx_combin), dtype=torch.float, device=self.__device),
            torch.ones(len(batch_states), dtype=torch.float, device=self.__device)
        ))
        # Concatenate all the tensors
        border = torch.vstack((expanded_pair_idx, expanded_times, expanded_states, border_marked_idx, is_edge))
        # Firstly, sort with respect to the pair indices, then time and finally states.
        border = border[:, border[2].argsort(stable=True)]
        border = border[:, border[1].argsort(stable=True)]
        border = border[:, border[0].argsort(stable=True)]

        expanded_pair_idx, expanded_times, _, border_marked_idx, is_edge = border

        # Compute the expanded states
        border_cum_sum = border_marked_idx.cumsum(0) - 1
        counts = torch.bincount(border_cum_sum.to(torch.long))
        expanded_states = torch.repeat_interleave(border[2][border_marked_idx == 1], counts).to(torch.long)

        # Compute the delta time
        delta_t = expanded_times[1:] - expanded_times[:-1]
        mask = (delta_t < 0)
        delta_t[mask] = self.__bin_bounds[-1] + delta_t[mask]
        delta_t = torch.concat((delta_t, (self.__bin_bounds[-1] - expanded_times[-1]).unsqueeze(0)))

        # If bin_idx is provided, filter the batch edges, edge times and states
        if bin_idx is not None:
            mask = (self.__bin_bounds[bin_idx] <= expanded_times) & (expanded_times < self.__bin_bounds[bin_idx+1])
            expanded_pair_idx = expanded_pair_idx[mask]
            expanded_times = expanded_times[mask]
            expanded_states = expanded_states[mask]
            is_edge = is_edge[mask]
            delta_t = delta_t[mask]

            delta_t[expanded_times == self.__bin_bounds[bin_idx+1]] = 0
            # print(sum( expanded_times + delta_t > self.__bin_bounds[bin_idx+1]))
            # idx = expanded_times + delta_t > self.__bin_bounds[bin_idx+1]
            # print( expanded_times[idx] )
            # print(delta_t[idx] )

        # Convert the linear indices to matrix indices
        expanded_pairs = utils.linearIdx2matIdx(expanded_pair_idx, n=self.__nodes_num, directed=self.__directed)

        return batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge.to(torch.bool), delta_t
