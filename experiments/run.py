import os
import torch
from src.dataset import Dataset
from src.learning import LearningModel
from argparse import ArgumentParser, RawTextHelpFormatter

# Global control for device
CUDA = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_arguments():
    """
    Parse the command line arguments
    """
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--edges', type=str, required=True, help='Path of the edge list file'
    )
    parser.add_argument(
        '--directed', type=bool, required=False, help='Directed or undirected graph'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--log_path', type=str, required=False, default=None, help='Path of the log file'
    )
    parser.add_argument(
        '--bins_num', type=int, default=100, required=False, help='Number of bins'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--init_time', type=int, required=False, help='The initial time point of the training dataset'
    )
    parser.add_argument(
        '--last_time', type=int, required=False, help='The last time point of the training dataset'
    )
    parser.add_argument(
        '--k', type=int, default=10, required=False, help='Latent dimension size of the prior element'
    )
    parser.add_argument(
        '--prior_lambda', type=float, default=None, required=False, help='Scaling coefficient of the covariance'
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
        '--lr', type=float, default=0.01, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--device', type=str, default="gpu", required=False, help='Device'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )
    return parser.parse_args()


def train(dataset: Dataset, nodes_num: int, bins_num: int, dim: int, k, directed, 
          prior_lambda: float, lr: float, batch_size: int, epoch_num: int, steps_per_epoch: int, 
          device: torch.device, verbose: bool, seed: int, model_path: str, log_path: str):

    # Define the learning model
    lm = LearningModel(
        nodes_num=nodes_num, directed=directed, bins_num=bins_num, dim=dim, k=k,
        prior_lambda=prior_lambda, device=device, verbose=verbose, seed=seed,
    )
    # Learn the hyper-parameters
    lm.learn(dataset=dataset, lr=lr, batch_size=batch_size, epoch_num=epoch_num, steps_per_epoch=steps_per_epoch)
    # Save the model
    lm.save(path=model_path)


def process(parser):

    args = {
        'model_path': parser.model_path, 'log_path': parser.log_path, 'directed': parser.directed, 
        'bins_num': parser.bins_num, 'dim': parser.dim, 'prior_lambda': parser.prior_lambda,  'k': parser.k,  
        'epoch_num': parser.epoch_num, 'steps_per_epoch': parser.spe, 'batch_size': parser.batch_size, 'lr': parser.lr, 
        'device': parser.device, 'seed': parser.seed, 'verbose': parser.verbose,
    }

    # Load the dataset
    dataset = Dataset()
    dataset.read_edgelist(parser.edges)
    if parser.init_time is not None:
        dataset.set_init_time(parser.init_time)
    if parser.last_time is not None:
        dataset.set_last_time(parser.last_time)
    nodes_num = dataset.get_nodes_num()

    # If the prior lambda is given, then train the model with this value
    if args['prior_lambda'] is not None:
        train(dataset=dataset, nodes_num=nodes_num, **args) 

    # Otherwise, find the best prior lambda by applying the aneealing strategy over the validation set
    else:
        for current_prior_lambda in [1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6]:
            args['prior_lambda'] = current_prior_lambda
            train(dataset=dataset, nodes_num=nodes_num, **args) 


if __name__ == '__main__':
    parser = parse_arguments()
    process(parser)

