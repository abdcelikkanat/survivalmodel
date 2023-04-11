import unittest
import torch
import utils


class TestUtils(unittest.TestCase):

    def test_pair_iter_undirected(self):

        n = 10

        pairs = [(i, j) for i, j in utils.pair_iter(n, directed=False)]
        ground_truth = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
            (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
            (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
            (5, 6), (5, 7), (5, 8), (5, 9),
            (6, 7), (6, 8), (6, 9),
            (7, 8), (7, 9),
            (8, 9)
        ]

        self.assertSequenceEqual(pairs, ground_truth)

    def test_pair_iter_directed(self):

        n = 10

        pairs = [(i, j) for i, j in utils.pair_iter(n, directed=True)]
        ground_truth = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
            (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
            (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
            (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
            (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9),
            (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 8), (7, 9),
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 9),
            (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8),
        ]

        self.assertSequenceEqual(pairs, ground_truth)

    def test_pairIdx2linearIdx_undirected(self):

        n = 10

        linearIdx = [utils.pairIdx2flatIdx(i=i, j=j, n=n, directed=False) for i, j in utils.pair_iter(n, directed=False)]
        ground_truth = list(range(n*(n-1)//2))

        self.assertSequenceEqual(linearIdx, ground_truth)

    def test_linearIdx2pairIdx_undirected(self):

        n = 10

        pairIdx = list(map(tuple, utils.linearIdx2matIdx(idx=torch.arange(n*(n-1)//2), n=n, directed=False).tolist()))
        ground_truth = list(map(tuple, utils.pair_iter(n, directed=False)))

        self.assertSequenceEqual(pairIdx, ground_truth)

    def test_pairIdx2linearIdx_directed(self):

        n = 10

        linearIdx = [utils.pairIdx2flatIdx(i=i, j=j, n=n, directed=True) for i, j in utils.pair_iter(n, directed=True)]
        ground_truth = list(range(n*(n-1)))

        self.assertSequenceEqual(linearIdx, ground_truth)
    
    def test_linearIdx2pairIdx_directed(self):

        n = 10

        pairIdx = list(map(tuple, utils.linearIdx2matIdx(idx=torch.arange(n*(n-1)), n=n, directed=True).tolist()))
        ground_truth = list(map(tuple, utils.pair_iter(n, directed=True)))

        self.assertSequenceEqual(pairIdx, ground_truth)

if __name__ == '__main__':
    unittest.main()