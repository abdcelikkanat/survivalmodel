import os
import torch
import numpy as np
import pickle
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


bins_num = 100
dim = 2
last_time = 1.0
epoch_num = 300 #510
steps_per_epoch = 1 #10
batch_size = 0 #50 #-1
lr = 0.01 #0.01
seed = 19 #19
verbose = True
k=10
prior_lambda=1e6
directed = False


# Define the dataset name
dataset_name = "synthetic_2clusters_" + ('directed' if directed else 'undirected')
# Get the current folder path
current_folder = os.path.dirname(os.path.abspath(__file__))
# Define the dataset folder path
dataset_folder=os.path.join(current_folder, '..', 'datasets', dataset_name)
# Define the data path
data_path = os.path.join(dataset_folder, f"{dataset_name}.edges")
# Define the model name
model_name = f"{dataset_name}_B={bins_num}_D={dim}_lr={lr}_ep={epoch_num}_lambda={prior_lambda}"
# Define the model path for saving
model_folder = os.path.join(current_folder, '..', 'models')
# Define the model path
model_path = os.path.join(model_folder, model_name + ".model")

dataset = Dataset()
dataset.read_edgelist(data_path)
nodes_num = dataset.get_nodes_num()

print("Nodes num", nodes_num)
print(f"Min time: {max(dataset.get_edge_times())}")


# Define the learning model
lm = LearningModel(
    data=dataset, nodes_num=nodes_num, bins_num=bins_num, dim=dim, directed=directed,
    prior_lambda=prior_lambda,
    lr=lr, batch_size=batch_size, epoch_num=epoch_num, steps_per_epoch=steps_per_epoch,
    device=torch.device(device), verbose=verbose, seed=seed,
)
lm.learn()
lm.save(path=model_path)
# # torch.save({'model_state_dict': lm.state_dict(),}, model_path)

number_of_frames = 100
frame_times = torch.linspace(0, last_time, steps=number_of_frames)
nodes=torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, number_of_frames)
time_list = frame_times.unsqueeze(0).expand(nodes_num, number_of_frames)

rt_s = lm.get_rt_s(time_list=time_list.flatten(), nodes=nodes.flatten())
rt_r = lm.get_rt_r(time_list=time_list.flatten(), nodes=nodes.flatten()) if directed else None


embs_s = rt_s.reshape(nodes_num, number_of_frames, dim).transpose(0, 1).detach().numpy()
embs_r = rt_r.reshape(nodes_num, number_of_frames, dim).transpose(0, 1).detach().numpy() if directed else None
anim = Animation(
    embs=(embs_s, embs_r),
    data=(dataset.get_edges().T, dataset.get_edge_times()), directed=directed,
    fps=12,
    node2color={node:node for node in range(nodes_num)},
    frame_times=frame_times.detach().numpy()
)
anim.save(os.path.join(model_folder, f"{model_name}.mp4"), format="mp4")

