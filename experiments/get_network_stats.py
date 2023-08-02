from argparse import ArgumentParser, RawTextHelpFormatter
from src.dataset import Dataset

def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--edges', type=str, required=True, help='Path of the edge list file'
    )
    parser.add_argument(
        '--output_path', type=str, default=None, required=False, help='Path of the output file'
    )

    return parser.parse_args()


def process(parser):

    # Read the arguments
    edges_path = parser.edges

    # Load the dataset
    dataset = Dataset()
    dataset.read_edge_list(file_path=edges_path)
    dataset.print_info()

    if parser.output_path is not None:
        dataset.write_data(parser.output_path)


if __name__ == "__main__":
    args = parse_arguments()
    process(args)