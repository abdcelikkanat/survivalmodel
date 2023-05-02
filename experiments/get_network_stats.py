from argparse import ArgumentParser, RawTextHelpFormatter
from src.dataset import Dataset

def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--edges', type=str, required=True, help='Path of the edge list file'
    )

    return parser.parse_args()


def process(parser):

    # Read the arguments
    edges_path = parser.edges

    # Load the dataset
    dataset = Dataset()
    dataset.read_edgelist(edges_path)
    dataset.print_info()


if __name__ == "__main__":
    args = parse_arguments()
    process(args)