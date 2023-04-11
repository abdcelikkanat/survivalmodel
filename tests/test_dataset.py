import unittest
import torch
from src.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_split_over_time_nonisolated_nodes(self):

        edges = torch.as_tensor([(0, 1), (0, 4), (0, 5), (1, 3), (1, 2), (2, 3), (3, 4), (4, 5)], dtype=torch.long)
        edge_times = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.8], dtype=torch.float)
        init_time = 0.0
        last_time = 1.0

        ds = Dataset(
            nodes_num = len(torch.unique(edges)), edges=edges, edge_times=edge_times, init_time=init_time, last_time=last_time
        )

        first_split, second_split = ds.split_over_time(split_ratio=0.5)
        first_edges, second_edges = list(first_split.get_edges()), list(second_split.get_edges())

        self.assertSequenceEqual(first_edges, list(zip(map(tuple, edges[:5].tolist()), edge_times[:5].tolist())))
        self.assertSequenceEqual(second_edges, list(zip(map(tuple, edges[5:].tolist()), edge_times[5:].tolist())))
        

    def test_split_over_time_isolated_nodes(self):

        edges = torch.as_tensor([(0, 1), (1, 2), (2, 3), (1, 3), (1, 4)], dtype=torch.long)
        edge_times = torch.as_tensor([0.1, 0.2, 0.3, 0.7, 0.8], dtype=torch.float)
        init_time = 0.0
        last_time = 1.0

        ds = Dataset(
            nodes_num = len(torch.unique(edges)), edges=edges, edge_times=edge_times, init_time=init_time, last_time=last_time
        )

        first_split, second_split = ds.split_over_time(split_ratio=0.5)
        first_edges, second_edges = list(first_split.get_edges()), list(second_split.get_edges())
        
        self.assertSequenceEqual(first_edges, list(zip(map(tuple, edges[:3].tolist()), edge_times[:3].tolist())))
        self.assertSequenceEqual(second_edges, list(zip(map(tuple, edges[3:4].tolist()), edge_times[3:4].tolist())))

    def test_split_over_time_isolated_nodes_and_relabel(self):

        edges = torch.as_tensor([(0, 1), (0, 3), (0, 4), (2, 3), (0, 3), (1, 4)], dtype=torch.long)
        edge_times = torch.as_tensor([0.1, 0.3, 0.4, 0.7, 0.8, 0.9], dtype=torch.float)
        init_time = 0.0
        last_time = 1.0

        ds = Dataset(
            nodes_num = len(torch.unique(edges)), edges=edges, edge_times=edge_times, init_time=init_time, last_time=last_time
        )

        first_split, second_split = ds.split_over_time(split_ratio=0.5)
        first_edges, second_edges = list(first_split.get_edges()), list(second_split.get_edges())

        self.assertSequenceEqual(
            first_edges, 
            [((0, 1), 0.10000000149011612), ((0, 2), 0.30000001192092896), ((0, 3), 0.4000000059604645)]
        )
        self.assertSequenceEqual(
            second_edges, 
            [((0, 2), 0.800000011920929), ((1, 3), 0.8999999761581421)]
        )

if __name__ == '__main__':
    unittest.main()