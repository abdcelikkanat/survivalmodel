import unittest
import torch
from src.base import BaseModel


class TestBase(unittest.TestCase):

    def test_delta_rt(self):

        x0_s = torch.as_tensor(
            [[0., 0.], [0., 0.]]
        )
        v_s = torch.as_tensor(
            [
                [[1., 0.], [-1., 0.]],
                [[-1., 0.], [1., 0.]]
            ]
        )
        beta_s = torch.zeros(size=(2, ), dtype=torch.float)

        bm = BaseModel(x0_s=x0_s, v_s=v_s, beta_s=beta_s)
        delta_rt = bm.get_delta_rt(
            time_list=torch.as_tensor([0, 0.25, 0.5, 0.75, 1.]),
            pairs=torch.as_tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        )

        self.assertSequenceEqual(
            delta_rt.tolist(),
            [
                [0.0000, 0.0000],
                [0.5000, 0.0000],
                [1.0000, 0.0000],
                [0.5000, 0.0000],
                [0.0000, 0.0000]
            ]
        )


if __name__ == '__main__':
    unittest.main()