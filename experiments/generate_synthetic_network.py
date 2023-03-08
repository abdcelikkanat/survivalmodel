import os
import torch
from src.dataset import Dataset
from src.construction import InitialPositionVelocitySampler, ConstructionModel
from src.animation import Animation

#  Sample the intial position and velocity vectors
dataset_name = "synthetic_feb12_75e-2"
dim = 2
bins_num = 100
cluster_sizes = [15]*4
nodes_num = sum(cluster_sizes)
prior_lambda = 75e-2
prior_sigma = 0.1
prior_B_x0_c = 1.0
prior_B_ls = 1.0
device = torch.device = "cpu"
verbose = True
seed = 19
normalize=False


# Get the current folder path
current_folder = os.path.dirname(os.path.abspath(__file__))

# Define the dataset folder path
dataset_folder=os.path.join(current_folder, '..', 'datasets', dataset_name)

# Define the animation path
anim_path = os.path.join(dataset_folder, "ground_truth.mp4")

# If the dataset folder does not exist, create it
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Define the sampler
ipv = InitialPositionVelocitySampler(
    dim=dim, bins_num=bins_num, cluster_sizes=cluster_sizes, 
    prior_lambda=prior_lambda, prior_sigma=prior_sigma, prior_B_x0_c = prior_B_x0_c, prior_B_ls=prior_B_ls,
    device = device, verbose = verbose, seed=seed
)

# Set the bias terms to zero
beta = torch.randn(size=(nodes_num, 2), dtype=torch.float)
init_states = torch.zeros(size=(nodes_num * (nodes_num - 1)//2, ), dtype=torch.bool).bool()
x0, v, last_time = ipv.sample()

cm = ConstructionModel(
    x0=x0, v=v, beta=beta, last_time=last_time, bins_num=bins_num, init_states=init_states, seed=seed
)

# Save the generated dataset
cm.save(folder_path=dataset_folder, normalize=normalize)

# Load the generated dataset
dataset = Dataset(path=dataset_folder, seed=seed, normalize=normalize)
pairs, events = dataset.get_pairs(), dataset.get_events() #torch.asarray(dataset.get_pairs()).type(torch.bool), torch.asarray(dataset.get_events())

# Generate the animation
frame_times = torch.linspace(0, last_time, 100)
node2group = [c for c in range(len(cluster_sizes)) for _ in range(cluster_sizes[c])]
embs = cm.get_rt(
    time_list=frame_times.repeat_interleave(nodes_num),
    nodes=torch.arange(nodes_num).repeat(len(frame_times))
).reshape((len(frame_times), cm.get_number_of_nodes(), cm.get_dim())).detach().numpy()

anim = Animation(
    embs, data=(pairs, events),
    fps=12,
    node2color=node2group,
    frame_times=frame_times.numpy()
)
anim.save(anim_path, format="mp4")
    