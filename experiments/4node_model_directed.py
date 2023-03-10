import os
import torch
import numpy as np
from src.dataset import Dataset
from src.learning import LearningModel
from src.animation import Animation

# Global control for device
CUDA = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')




# dataset_name = "synthetic_jan18"
dataset_name = "4node"

bins_num = 100
dim = 2
last_time = 1.0
epoch_num = 120
steps_per_epoch = 1 #10
batch_size = 4 #50 #-1
lr = 0.1
normalize = True

seed = 19 #19
verbose = True

# Define the model name
model_name = f"{dataset_name}_B={bins_num}_D={dim}_T={last_time}_lr={lr}_bs={batch_size}_ep={epoch_num}_spe={steps_per_epoch}"

# Get the current folder path
current_folder = os.path.dirname(os.path.abspath(__file__))
# Define the model path for saving
model_path=os.path.join(current_folder, '..', 'models', model_name + ".model")

# Get the number of nodes
nodes_num = 4

# # data = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], [[], [0.1, 0.3], [], [], [0.7, 0.9], []]
#####
# pairs = [ (0, 2), (0, 2), (1, 3), (1, 3) ]
# events = [ 0.1,  0.6,  0.7, 0.9 ]
# states = [ 0, 0, 0, 0 ]
pairs = [ (0, 2), (0, 2), (1, 3), (1, 3), (2 ,3), (2, 3), (0, 3), (0, 3), (0, 1), (0, 1) ]
events = [ 0.1,  0.7,  0.2, 0.6, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5 ]
init_states = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

data = pairs, events, init_states
directed = False #True #False

# Define the learning model
lm = LearningModel(
    data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, directed=directed,
    lr=lr, batch_size=batch_size, epoch_num=epoch_num, steps_per_epoch=steps_per_epoch,
    device=torch.device(device), verbose=verbose, seed=seed,
)
lm.learn()

number_of_frames = 100
frame_times = torch.linspace(0, last_time, steps=number_of_frames)
nodes=torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, number_of_frames)
time_list = frame_times.unsqueeze(0).expand(nodes_num, number_of_frames)

rt_s = lm.get_rt_s(time_list=time_list.flatten(), nodes=nodes.flatten())
# rt_r = lm.get_rt_r(time_list=time_list.flatten(), nodes=nodes.flatten())


embs = rt_s.reshape(nodes_num, number_of_frames, dim).transpose(0, 1).detach().numpy()
anim = Animation(
    embs,
    data=(data[0], data[1]),
    fps=12,
    node2color={node:node for node in range(nodes_num)},
    frame_times=frame_times.detach().numpy()
)
anim.save(f"./emb_s_{dataset_name}_train_v_x0_seed={seed}.mp4", format="mp4")



# Save the model
# lm.save(model_path)

# from utils.common import *

# number_of_frames = 3
# print(
#     lm.get_rt(
#         time_list=torch.linspace(0, last_time, steps=number_of_frames).unsqueeze(0).expand(nodes_num, number_of_frames).flatten(), 
#         nodes=torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, number_of_frames).flatten()
#     )
# )

# print(
#     erfi(torch.as_tensor([2.0]))
# )

# t = torch.double
# print(
#     torch.special.erf(torch.complex(real=torch.as_tensor([2.0], dtype=t), imag=torch.as_tensor([0.0], dtype=t)))
# )

# import numpy as np
# import scipy.special as sp
# import cmath

# # Approximate the erfi function for the small real values
# # def approximated_erfi(z):

    
# #     # coeff = torch.as_tensor([1.0, 1./3, 1./5, 1./21, 1./108, 1./660, 1./4680, 1./37800, 1./342720, 1./3447360, 1./38102400, 1./459459200], dtype=torch.float)

# #     return 1./np.sqrt(np.pi) * ( 
# #         2*z + 2./3 *(z**3) + 1./5*(z**5) + 1./21*(z**7) + 1./108*(z**9) + 1./660*(z**11) + 
# #         1.0/4680*(z**13) + 1./37800*(z**15) + 1./342720*(z**17) + 1./3447360*(z**19) + 1./38102400*(z**21) + 1./459459200*(z**23) +
# #         1.0/5987520000*(z**25) 
# #         )

# def approximate_erf(z):
#     '''
#     Approximate the erf function with maximum error of 1.5e-7, 
#     Source: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables by Abramowitz M. and Stegun I.A.
#             Equation 7.1.26
#     '''
#     p = .3275911
#     a1 = .254829592
#     a2 = -.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429

#     t = 1.0 / (1.0 + p*z)

#     return 1 - (t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))) * np.exp(-z**2)

# def approximated_erfi(z):
#     '''
#     Approximate the erf function with maximum error of 1.5e-7, 
#     Source: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables by Abramowitz M. and Stegun I.A.
#             Equation 7.1.26
#     '''
#     p = .3275911
#     a1 = .254829592
#     a2 = -.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429

#     # t = 1.0 / (1.0 + p*z)
#     # t = 1.0 / (1.0 + p*complex(0, z))
#     # t = (1 + 1j*-p*z)  / (1.0 + (p*z)**2)

#     # z_ = 1j*z
#     # t = 1.0 / (1.0 + p*z_)
#     # return (-1j) * ( 1 - (t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))) * np.exp(-z_**2) )

#     t = z
#     a = .36
#     c0 = a / np.sqrt(np.pi)
#     return c0 * (1. + 2*sum([np.exp(-(a*n)**2)*np.cosh(2*a*n*t) for n in range(1, 20)]))

# z = np.arange(0, 4, 1e-3)

# # sp_out = sp.erf(z)
# # approx_out = approximate_erf(z) 
# sp_out = sp.erfi(z)
# approx_out = approximated_erfi(z)

# # plot the sp out
# import matplotlib.pyplot as plt
# plt.plot(z, sp_out, label="scipy")
# plt.plot(z, approx_out, label="approx")
# # plot the legends
# plt.legend()
# plt.show()

# # z = torch.as_tensor(z, dtype=torch.float)
# # print(z.shape)
# # print(
# #    torch.pow(z, torch.as_tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]) ) #.repeat(2, 1),# , #torch.as_tensor([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
    
# # )





# import numpy as np
# import scipy.special as sp
# import cmath
# import matplotlib.pyplot as plt

# def approximated_erfi(z: torch.Tensor):
#     '''
#     Approximate the erf function with maximum error of 1.5e-7, 
#     Source: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables by Abramowitz M. and Stegun I.A.
#             Equation 7.1.26
#     '''
#     p = .3275911
#     a1 = .254829592
#     a2 = -.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429
    
#     z_ = 1j * z
#     t = 1.0 / (1.0 + p*z_)

#     return ( 1 - (t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))) * torch.exp(-z_**2) ).imag


# z = np.arange(0, 4, 1e-3)
# sp_out = sp.erfi(z)
# approx_out = approximated_erfi(torch.asarray(z)).numpy()

# # plot the sp out
# plt.plot(z, sp_out, label="scipy")
# plt.plot(z, approx_out, label="approx")
# # plot the legends
# plt.legend()
# plt.show()