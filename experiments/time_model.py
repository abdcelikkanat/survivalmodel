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
dataset_name = "time"

bins_num = 100
dim = 2
last_time = 1.0
epoch_num = 10000
steps_per_epoch = 1 #10
batch_size = 4 #50 #-1
learning_rate = 0.1
normalize = True

seed = 19 #19
verbose = True

# Define the model name
model_name = f"{dataset_name}_B={bins_num}_D={dim}_T={last_time}_lr={learning_rate}_bs={batch_size}_ep={epoch_num}_spe={steps_per_epoch}"

# Get the current folder path
current_folder = os.path.dirname(os.path.abspath(__file__))
# Define the model path for saving
model_path=os.path.join(current_folder, '..', 'models', model_name + ".model")

# Get the number of nodes
nodes_num = 4

# # data = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], [[], [0.1, 0.3], [], [], [0.7, 0.9], []]
pairs = [] #[ (0, 2), (0, 2), (1, 3), (1, 3) ]
events = [] #[ 0.1, 0.9,  0.7, 0.9 ]
data = pairs, events


dataset = Dataset(edge_file='datasets/deneme.edges')
print(
    dataset.get_edges(), dataset.get_min_value(), dataset.get_max_value(),
)

# # Define the learning model
# lm = LearningModel(
#     data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=last_time,
#     masked_pairs=None,
#     learning_rate=learning_rate, batch_size=batch_size, epoch_num=epoch_num, steps_per_epoch=steps_per_epoch,
#     device=torch.device(device), verbose=verbose, seed=seed,
# )
# lm.learn()

# number_of_frames = 100
# frame_times = torch.linspace(0, last_time, steps=number_of_frames)
# nodes=torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, number_of_frames)
# time_list = frame_times.unsqueeze(0).expand(nodes_num, number_of_frames)
# rt = lm.get_rt(time_list=time_list.flatten(), nodes=nodes.flatten())

# embs = rt.reshape(nodes_num, number_of_frames, dim).transpose(0, 1).detach().numpy()
# anim = Animation(
#     embs,
#     data=(data[0], data[1]),
#     fps=12,
#     node2color={node:node for node in range(nodes_num)},
#     frame_times=frame_times.detach().numpy()
# )
# anim.save(f"./emb_{dataset_name}_train_v_x0_seed={seed}.mp4", format="mp4")



# Save the model
# lm.save(model_path)
