import torch
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter
from src.learning import LearningModel
from src.animation import Animation
from src.dataset import Dataset
from utils.common import set_seed

# Global control for device
CUDA = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--edges', type=str, required=True, help='Path of the edge list file'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--anim_path', type=str, required=True, default="", help='Animation path'
    )
    parser.add_argument(
        '--frames_num', type=int, required=False, default=100, help='Number of frames'
    )
    parser.add_argument(
        '--format', type=str, required=False, choices=["mp4", "gif"], default="mp4", help='Animation file format'
    )
    parser.add_argument(
        '--fps', type=int, required=False, default=12, help='Frame per second'
    )

    parser.add_argument(
        '--mask_path', type=str, required=False, default="", help='Path of the file storing node pairs for masking'
    )
    parser.add_argument(
        '--log', type=str, required=False, default=None, help='Path of the log file'
    )
    parser.add_argument(
        '--bins_num', type=int, default=100, required=False, help='Number of bins'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--last_time', type=float, default=1.0, required=False, help='The last time point of the training dataset'
    )
    parser.add_argument(
        '--k', type=int, default=10, required=False, help='Latent dimension size of the prior element'
    )
    parser.add_argument(
        '--prior_lambda', type=float, default=1e5, required=False, help='Scaling coefficient of the covariance'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=100, required=False, help='Number of epochs'
    )
    parser.add_argument(
        '--spe', type=int, default=1, required=False, help='Number of steps per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=0, required=False, help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )

    return parser.parse_args()


def process(parser):

    # Read the arguments
    edges_path = parser.edges
    model_path = parser.model_path
    anim_path = parser.anim_path
    file_format = parser.format
    fps = parser.fps

    # Load the dataset
    dataset = Dataset()
    dataset.read_edgelist(edges_path)
    nodes_num = dataset.get_nodes_num()

    # Load the model
    kwargs, lm_state = torch.load(model_path, map_location=torch.device(device))
    lm = LearningModel(**kwargs, device="cpu")
    lm.load_state_dict(lm_state)

    # Get the dimension size and directed flag
    dim = kwargs['dim']
    directed = kwargs['directed']

    init_time = dataset.get_init_time()
    last_time = dataset.get_last_time()
    ####
    number_of_frames = 100
    frame_times = torch.linspace(init_time, last_time, steps=number_of_frames)
    nodes=torch.arange(nodes_num).unsqueeze(1).expand(nodes_num, number_of_frames)
    time_list = frame_times.unsqueeze(0).expand(nodes_num, number_of_frames)

    rt_s = lm.get_rt_s(time_list=(time_list.flatten() - init_time ) / float(last_time - init_time), nodes=nodes.flatten())
    rt_s = rt_s.reshape(nodes_num, number_of_frames, dim).transpose(0, 1).detach().numpy()
    
    
    if directed:
        rt_r = lm.get_rt_r(time_list=(time_list.flatten() - init_time )/ float(last_time - init_time), nodes=nodes.flatten()) 
        rt_r = rt_r.reshape(nodes_num, number_of_frames, dim).transpose(0, 1).detach().numpy() 
    else:
        rt_r = None

    anim = Animation(
        embs=(rt_s, rt_r),
        frame_times=frame_times.detach().numpy(),
        dataset=dataset,
        # data=(dataset.get_edges().T, dataset.get_edge_times()), 
        directed=directed,
        fps=fps,
        node2color={node:node for node in range(nodes_num)},
    )
    anim.save(anim_path, format=file_format)
    


if __name__ == "__main__":
    args = parse_arguments()
    process(args)