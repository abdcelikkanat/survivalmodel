import unittest
from src.base import BaseModel
import torch

x0 = torch.as_tensor(
    [[0., 0.], [1., 0.]]
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


# time_list = torch.as_tensor([0., 1, 1.5, 2])
# x0 = torch.as_tensor(
#     [[0., 0.], [1., 1.]]
# )
# v = torch.as_tensor()

# rt = bm.get_rt(time_list=time_list, nodes=torch.as_tensor([0, 1, 1, 1]), standardize=False)
# print(rt)
# print(xt) tensor([[0.0000, 0.0000],
#         [0.0000, 0.0000],
#         [0.5000, 0.0000],
#         [1.0000, 0.0000]])

#####################################################################################################################
time_list = torch.as_tensor([0., 0.5], dtype=torch.float)
pairs = torch.as_tensor([[0, 0], [1, 1]], dtype=torch.long)
delta_t = torch.as_tensor([0.1, 0.2], dtype=torch.float)
states = torch.as_tensor([0, 1], dtype=torch.long)
rt = bm.get_intensity_integral(time_list=time_list, pairs=pairs, delta_t=delta_t, states=states, standardize=False)
print(rt)
#####################################################################################################################



#####################################################################################################################

#####################################################################################################################