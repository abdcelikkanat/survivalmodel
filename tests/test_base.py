import unittest
from src.base import BaseModel
import torch

class XTestBase(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def text_delta_xt():

        x0 = torch.as_tensor(
            [[0., 0.]]
        )
        v = torch.as_tensor(
            [
                [[1., 0.], [-1., 0.]],
                [[-1., 0.], [1., 0.]]
            ]
        )
        init_states = torch.zeros(size=(2, ), dtype=torch.float)
        beta0 = torch.zeros(size=(2, ), dtype=torch.float) 
        beta1 = torch.zeros(size=(2, ), dtype=torch.float)
        bins_num = v.shape[0]
        last_time = 2.0

        bm = BaseModel(x0=x0, v=v, init_states=init_states, beta0=beta0, beta1=beta1, bins_num=bins_num, last_time=last_time)
        
        
        time_list = torch.as_tensor([0., 1.])
        # x0 = torch.as_tensor(
        #     [[0., 0.], [1., 1.]]
        # )
        # v = torch.as_tensor()

        xt = bm.get_delta_xt(time_list=time_list, x0=x0, v=v)
        print(xt)


if __name__ == '__main__':
    unittest.main()