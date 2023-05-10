import unittest
import torch
from src.sampler import BatchSampler

SEED = 19


class TestSampler(unittest.TestCase):

    def test_positive_3bins(self):

        bins_num = 3
        bin_bounds = torch.linspace(0, 1.0, bins_num + 1)

        edges = torch.as_tensor([
            [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2],
        ], dtype=torch.long).T
        edge_times = torch.as_tensor([
            0, 0, 0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8
        ], dtype=torch.float)
        edge_states = torch.as_tensor([
            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1
        ], dtype=torch.long)
        bs = BatchSampler(
            edges=edges, edge_times=edge_times, edge_states=edge_states,
            bin_bounds=bin_bounds, nodes_num=len(torch.unique(edges)), batch_size=0, directed=False, seed=SEED
        )

        batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge, delta_t = bs.sample()

        self.assertSequenceEqual(batch_nodes, [0, 1, 2])

        self.assertSequenceEqual(expanded_pairs.T.tolist(), [[0, 1]] * 7 + [[0, 2]] * 7 + [[1, 2]] * 7)

        self.assertSequenceEqual(
            expanded_times.tolist(),
            [0.0, 0.0, 0.20000000298023224, 0.3333333432674408, 0.5, 0.6666666269302368, 0.800000011920929] * 3,
        )

        self.assertSequenceEqual(expanded_states.tolist(), [0, 0, 1, 0, 0, 0, 0] * 3)

        self.assertSequenceEqual(is_edge.tolist(), [0, 1, 1, 0, 1, 0, 1] * 3)

        self.assertSequenceEqual(
            delta_t.tolist(),
            [0.0, 0.20000000298023224, 0.13333334028720856, 0.1666666567325592,
             0.16666662693023682, 0.13333338499069214, 0.19999998807907104] * 3
        )

    def test_positive_4bins(self):

        bins_num = 4
        bin_bounds = torch.linspace(0, 1.0, bins_num + 1)

        edges = torch.as_tensor([
            [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2],
        ], dtype=torch.long).T
        edge_times = torch.as_tensor([
            0, 0, 0, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0
        ], dtype=torch.float)
        edge_states = torch.as_tensor([
            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1
        ], dtype=torch.long)
        bs = BatchSampler(
            edges=edges, edge_times=edge_times, edge_states=edge_states,
            bin_bounds=bin_bounds, nodes_num=len(torch.unique(edges)), batch_size=0, directed=False, seed=SEED
        )

        batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge, delta_t = bs.sample()

        self.assertSequenceEqual(batch_nodes, [0, 1, 2])

        self.assertSequenceEqual(expanded_pairs.T.tolist(), [[0, 1]] * 8 + [[0, 2]] * 8 + [[1, 2]] * 8)

        self.assertSequenceEqual(
            expanded_times.tolist(),
            [0.0, 0.0, 0.25, 0.30000001192092896, 0.5, 0.75, 0.800000011920929, 1.0] * 3,
        )

        self.assertSequenceEqual(expanded_states.tolist(), [0, 0, 1, 1, 0, 0, 0, 0] * 3)

        self.assertSequenceEqual(is_edge.tolist(), [0, 1, 0, 1, 0, 0, 1, 1] * 3)

        self.assertSequenceEqual(
            delta_t.tolist(),
            [0., 0.25, 0.050000011920928955, 0.19999998807907104, 0.25, 0.050000011920928955, 0.19999998807907104, 0.]*3
        )

    def test_negative_4bins(self):

        bins_num = 4
        bin_bounds = torch.linspace(0, 1.0, bins_num + 1)

        edges = torch.as_tensor([
            [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2],
        ], dtype=torch.long).T
        edge_times = torch.as_tensor([
            0, 0, 0, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0
        ], dtype=torch.float)
        edge_states = torch.as_tensor([
            -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1
        ], dtype=torch.long)
        bs = BatchSampler(
            edges=edges, edge_times=edge_times, edge_states=edge_states,
            bin_bounds=bin_bounds, nodes_num=len(torch.unique(edges)), batch_size=0, directed=False, seed=SEED
        )

        batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge, delta_t = bs.sample()

        self.assertSequenceEqual(batch_nodes, [0, 1, 2])

        self.assertSequenceEqual(expanded_pairs.T.tolist(), [[0, 1]] * 8 + [[0, 2]] * 8 + [[1, 2]] * 8)

        self.assertSequenceEqual(
            expanded_times.tolist(),
            [0.0, 0.0, 0.25, 0.30000001192092896, 0.5, 0.75, 0.800000011920929, 1.0] * 3,
        )
        self.assertSequenceEqual(expanded_states.tolist(), [0, 0, -1, -1, 0, 0, 0, 0] * 3)

        self.assertSequenceEqual(is_edge.tolist(), [0, 1, 0, 1, 0, 0, 1, 1] * 3)

        self.assertSequenceEqual(
            delta_t.tolist(),
            [0., 0.25, 0.050000011920928955, 0.19999998807907104, 0.25, 0.050000011920928955, 0.19999998807907104, 0.]*3
        )

    def test_mixed1_4bins(self):

        bins_num = 4
        bin_bounds = torch.linspace(0, 1.0, bins_num + 1)

        edges = torch.as_tensor([
            [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2],
        ], dtype=torch.long).T
        edge_times = torch.as_tensor([
            0, 0, 0, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0
        ], dtype=torch.float)
        edge_states = torch.as_tensor([
            -1, -1, -1, 1, 1, 1, 0, 0, 0, -1, -1, -1
        ], dtype=torch.long)
        bs = BatchSampler(
            edges=edges, edge_times=edge_times, edge_states=edge_states,
            bin_bounds=bin_bounds, nodes_num=len(torch.unique(edges)), batch_size=0, directed=False, seed=SEED
        )

        batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge, delta_t = bs.sample()

        self.assertSequenceEqual(batch_nodes, [0, 1, 2])

        self.assertSequenceEqual(expanded_pairs.T.tolist(), [[0, 1]] * 8 + [[0, 2]] * 8 + [[1, 2]] * 8)

        self.assertSequenceEqual(
            expanded_times.tolist(),
            [0.0, 0.0, 0.25, 0.30000001192092896, 0.5, 0.75, 0.800000011920929, 1.0] * 3,
        )
        self.assertSequenceEqual(expanded_states.tolist(), [0, 0, -1, -1, 1, 1, 1, 0] * 3)

        self.assertSequenceEqual(is_edge.tolist(), [0, 1, 0, 1, 0, 0, 1, 1] * 3)

        self.assertSequenceEqual(
            delta_t.tolist(),
            [0., 0.25, 0.050000011920928955, 0.19999998807907104, 0.25, 0.050000011920928955, 0.19999998807907104, 0.]*3
        )

    def test_mixed2_4bins(self):

        bins_num = 4
        bin_bounds = torch.linspace(0, 1.0, bins_num + 1)

        edges = torch.as_tensor([
            [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2],
        ], dtype=torch.long).T
        edge_times = torch.as_tensor([
            0, 0, 0, 0.3, 0.3, 0.3, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0
        ], dtype=torch.float)
        edge_states = torch.as_tensor([
            -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1
        ], dtype=torch.long)
        bs = BatchSampler(
            edges=edges, edge_times=edge_times, edge_states=edge_states,
            bin_bounds=bin_bounds, nodes_num=len(torch.unique(edges)), batch_size=0, directed=False, seed=SEED
        )

        batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge, delta_t = bs.sample()

        self.assertSequenceEqual(batch_nodes, [0, 1, 2])

        self.assertSequenceEqual(expanded_pairs.T.tolist(), [[0, 1]] * 8 + [[0, 2]] * 8 + [[1, 2]] * 8)

        self.assertSequenceEqual(
            expanded_times.tolist(),
            [0.0, 0.0, 0.25, 0.30000001192092896, 0.5, 0.75, 0.800000011920929, 1.0] * 3,
        )

        self.assertSequenceEqual(expanded_states.tolist(), [0, 0, -1, -1, 0, 0, 0, 1] * 3)

        self.assertSequenceEqual(is_edge.tolist(), [0, 1, 0, 1, 0, 0, 1, 1] * 3)

        self.assertSequenceEqual(
            delta_t.tolist(),
            [0., 0.25, 0.050000011920928955, 0.19999998807907104, 0.25, 0.050000011920928955, 0.19999998807907104, 0.]*3
        )


if __name__ == '__main__':
    unittest.main()