import torch
from src.dataset import Dataset
from src.learning import LearningModel
from argparse import ArgumentParser, RawTextHelpFormatter


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
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--init_model', type=str, required=False, default=None, help='Path of another model file for initialization'
    )
    parser.add_argument(
        '--mask_path', type=str, required=False, default=None, help='Path of the masked edge list file'
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
        '--prior_lambda', type=float, default=1e6, required=False, help='Scaling coefficient of the covariance'
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
        '--device', type=str, default="cuda", required=False, help='Device'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )
    return parser.parse_args()


def process(parser):

    # Load the dataset
    dataset = Dataset()
    dataset.read_edge_list(parser.edges)

    # Get the number of nodes
    nodes_num = dataset.get_nodes_num()
    # Check if the network is directed
    directed = dataset.is_directed()
    # Check if the network is signed
    signed = dataset.is_signed()

    # Read the masked pair file if exists
    if parser.mask_path is not None:
        masked_data = Dataset()
        masked_data.read_edge_list(parser.mask_path)
    else:
        masked_data = parser.mask_path

    # Print the information of the dataset
    if parser.verbose:
        dataset.print_info()

    # If the init model is not given, then train the model from scratch
    if parser.init_model is None:

        # Define the learning model
        lm = LearningModel(
            nodes_num=nodes_num, directed=directed, signed=signed, bins_num=parser.bins_num, dim=parser.dim,
            prior_lambda=parser.prior_lambda, device=parser.device, verbose=parser.verbose, seed=parser.seed,
        )

    else:

        # Load the model if exists
        kwargs, lm_state = torch.load(parser.init_model, map_location=torch.device(parser.device))
        # Update the arguments
        kwargs['seed'], kwargs['device'], kwargs['verbose'] = parser.seed, parser.device, parser.verbose

        # Load the model
        lm = LearningModel(**kwargs)
        lm.load_state_dict(lm_state)

    # Learn the hyper-parameters
    lm.learn(
        dataset=dataset, masked_data=masked_data, log_file_path=parser.log_path,
        lr=parser.lr, batch_size=parser.batch_size, epoch_num=parser.epoch_num, steps_per_epoch=parser.spe,
    )
    # Save the model
    lm.save(path=parser.model_path)


if __name__ == '__main__':
    parser = parse_arguments()
    process(parser)

