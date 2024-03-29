import os
import torch
from src.dataset import Dataset
from src.construction import ConstructionModel
from src.animation import Animation

#  Sample the intial position and velocity vectors
args = {
    'cluster_sizes': [5]*10, 'bins_num': 100, 'dim': 2, 'directed': False,   #[5]*4
    'beta_s':-0.1, 'beta_r': 0.,
    'prior_lambda': 1.0, #1.75e0 # Expand the space
    'prior_sigma_s': 1e-1,'prior_sigma_r': 8e-2,  #1e-1 #1e-2, 
    'prior_B_x0_logit_c_s': 1.5, 'prior_B_x0_logit_c_r': 1e+0, #(1e0, 1e+0)
    'prior_B_ls_s': 1e-1, 'prior_B_ls_r': 5e-1, #(1e-3, 1e-3) # controls how crazy the nodes are, craziness ~ 1./value
    'device': "cpu", 'verbose': True, 'seed': 1,
}

# Define the dataset name
dataset_name = f"synthetic_april21_{'directed' if args['directed'] else 'undirected'}"

# Define the dataset folder path
dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', dataset_name)
# Define the info path
info_path = os.path.join(dataset_folder, "info.txt")
# Define the animation path
anim_path = os.path.join(dataset_folder, "ground_truth.mp4")

# If the dataset folder does not exist, create it
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Initialize the constructor model
cm = ConstructionModel(**args)
# Save the generated dataset and model
cm.save_data(file_path=os.path.join(dataset_folder, 'data.pkl'))
cm.save_model(file_path=os.path.join(dataset_folder, 'data.model'))
cm.write_edges(file_path=os.path.join(dataset_folder, f"{dataset_name}.edges"))

bm = cm.get_model()
data = cm.get_data()

pairs, events, states, _, _, _ = data
init_time = events.min() #0 if len(data) <= 4 else data[4]
last_time = events.max() #1.0 if len(data) <= 5 else data[5]

# Animation
number_of_frames = 100
nodes_num = sum(args['cluster_sizes'])
frame_times = torch.linspace(init_time, last_time, steps=number_of_frames)
nodes=torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, number_of_frames)
time_list = frame_times.unsqueeze(0).expand(nodes_num, number_of_frames)

# Get the node representations for a given time list
rt_s = bm.get_rt_s(time_list=(time_list.flatten() - init_time)/(last_time - init_time), nodes=nodes.flatten())
rt_s = rt_s.reshape(nodes_num, number_of_frames, args['dim']).transpose(0, 1).detach().numpy()
if args['directed']:
    rt_r = bm.get_rt_r(time_list=(time_list.flatten() - init_time)/(last_time - init_time), nodes=nodes.flatten()) 
    rt_r = rt_r.reshape(nodes_num, number_of_frames, args['dim']).transpose(0, 1).detach().numpy()
else:
    rt_r = None

dataset = Dataset()
dataset.read_edgelist( os.path.join(dataset_folder, f"{dataset_name}.edges") )

anim = Animation(
    embs=(rt_s, rt_r),
    # data=(data[0], data[1]), 
    dataset=dataset,
    frame_times=frame_times.detach().numpy(),
    directed=args['directed'],
    fps=12,
    node2color={node:node for node in range(nodes_num)},
)
anim.save(anim_path, format="mp4")

# Save the info file
with open(info_path, 'w') as f:
    for key, value in args.items():
        f.write(f"{key}: {value}\n")


# dataset = Dataset(path=dataset_folder, seed=seed, normalize=normalize)
# pairs, events = dataset.get_pairs(), dataset.get_events() #torch.asarray(dataset.get_pairs()).type(torch.bool), torch.asarray(dataset.get_events())

# # Generate the animation
# frame_times = torch.linspace(0, last_time, 100)
# node2group = [c for c in range(len(cluster_sizes)) for _ in range(cluster_sizes[c])]
# embs = cm.get_rt(
#     time_list=frame_times.repeat_interleave(nodes_num),
#     nodes=torch.arange(nodes_num).repeat(len(frame_times))
# ).reshape((len(frame_times), cm.get_number_of_nodes(), cm.get_dim())).detach().numpy()

# anim = Animation(
#     embs, data=(pairs, events),
#     fps=12,
#     node2color=node2group,
#     frame_times=frame_times.numpy()
# )
# anim.save(anim_path, format="mp4")
    