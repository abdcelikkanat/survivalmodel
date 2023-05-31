import torch

import utils
from utils.common import flatIdx2matIdx


def expand_data(nodes_num: int, directed: bool, bin_bounds: torch.Tensor, edge_pair_flat_idx: torch.LongTensor,
                edge_times: torch.FloatTensor, edge_states: torch.LongTensor,
                border_pair_flat_idx: torch.LongTensor = None, device: torch.device = "cpu"):

    if border_pair_flat_idx is None:
        border_pair_flat_idx = torch.unique(edge_pair_flat_idx)

    bins_num = len(bin_bounds) - 1
    # Expand the batch edges, times and states with the bin border times
    expanded_pair_idx = torch.concat((
        border_pair_flat_idx.repeat_interleave(repeats=bins_num, dim=0),
        edge_pair_flat_idx
    ))
    expanded_times = torch.concat((
        bin_bounds[:-1].repeat(len(border_pair_flat_idx)),
        edge_times
    ))
    expanded_states = torch.concat((
        torch.zeros(bins_num * len(border_pair_flat_idx), dtype=torch.float, device=device),
        edge_states.to(torch.float)
    ))
    # Mark the first time of bin borders to detect later the pair edges
    border_marked_idx = torch.concat((
        torch.eye(1, bins_num, dtype=torch.float, device=device).squeeze(0).repeat(
            len(border_pair_flat_idx)),
        torch.ones(len(edge_states), dtype=torch.float, device=device)
    ))
    # An indicator vector to detect additional edge times
    is_edge = torch.concat((
        torch.zeros(bins_num * len(border_pair_flat_idx), dtype=torch.float, device=device),
        torch.ones(len(edge_states), dtype=torch.float, device=device)
    ))
    # Concatenate all the tensors
    border = torch.vstack((expanded_pair_idx, expanded_times, expanded_states, border_marked_idx, is_edge))
    # Firstly, sort with respect to the pair indices, then time and finally states.
    border = border[:, border[2].abs().argsort(stable=True)]
    border = border[:, border[1].argsort(stable=True)]
    border = border[:, border[0].argsort(stable=True)]

    expanded_pair_idx, expanded_times, _, border_marked_idx, is_edge = border

    # Compute the expanded states
    border_cum_sum = border_marked_idx.cumsum(0) - 1
    border_cum_sum[is_edge == 1] -= 1
    counts = torch.bincount(border_cum_sum.to(torch.long), minlength=len(edge_states) + len(border_pair_flat_idx))
    expanded_states = torch.repeat_interleave(border[2][border_marked_idx == 1], counts).to(torch.long)

    # Compute the delta time
    delta_t = expanded_times[1:] - expanded_times[:-1]
    # delta_t of the last event time of each group will be negative, so set them to 1.0 - t
    mask = (delta_t < 0)
    delta_t[mask] = bin_bounds[-1] + delta_t[mask]
    # Add the delta_t for the last time of the whole list
    delta_t = torch.concat((delta_t, (bin_bounds[-1] - expanded_times[-1]).unsqueeze(0)))

    # Convert the linear indices to matrix indices
    expanded_pairs = flatIdx2matIdx(expanded_pair_idx, n=nodes_num, is_directed=directed)

    # If there is a zero time and if it is an edge, then make it non-edge
    is_edge[expanded_times == 0] = 0

    return expanded_pairs, expanded_times, expanded_states, is_edge.to(torch.bool), delta_t
