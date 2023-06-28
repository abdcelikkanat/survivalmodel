import os
import pickle
import torch
from src.dataset import Dataset
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.common import set_seed

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--split_folder', type=str, required=True, help='Path of the split folder'
)
parser.add_argument(
    '--samples_folder', type=str, required=True, help='Path of the samples folder'
)
parser.add_argument(
    '--max_sample_size', type=int, required=False, default=10000, help='Maximum sample size'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

split_folder = args.split_folder
samples_folder = args.samples_folder
max_sample_size = args.max_sample_size
seed = args.seed

########################################################################################################################

# Set the seed value
set_seed(seed=seed)

########################################################################################################################

# Read the train dataset
dataset = Dataset()
dataset.read_edge_list(split_folder + "/first_half.edges")
dataset.print_info()
init_time, last_time = dataset.get_init_time(), dataset.get_last_time()
nodes_num = dataset.get_nodes_num()
is_directed = dataset.is_directed()
is_signed = dataset.is_signed()
del dataset

MIN_INTERVAL_SIZE = int(1e-2 * (last_time - init_time))

########################################################################################################################


def durations2samples(pos_durations, zero_durations, neg_durations, max_sample_size):
    print("Generating samples...")

    # Construct the zero samples
    perm = torch.randperm(len(zero_durations))
    zero_chosen_idx = perm[:min(max_sample_size, len(zero_durations), len(pos_durations) + len(neg_durations))].tolist()

    zero_pairs, zero_intervals, zero_labels = [], [], []
    for idx in zero_chosen_idx:
        zero_pairs.append((zero_durations[idx][0], zero_durations[idx][1]))
        zero_intervals.append((zero_durations[idx][2].item(), zero_durations[idx][3].item()))
        zero_labels.append(zero_durations[idx][4])

        if zero_durations[idx][3].item() - zero_durations[idx][2].item() < MIN_INTERVAL_SIZE:
            raise Exception("Zero interval size is less than the minimum interval size")

    pos_pairs, pos_intervals, pos_labels = [], [], []
    for idx in range(len(pos_durations)):
        pos_pairs.append((pos_durations[idx][0], pos_durations[idx][1]))
        pos_intervals.append((pos_durations[idx][2].item(), pos_durations[idx][3].item()))
        pos_labels.append(pos_durations[idx][4])

        if pos_durations[idx][3].item() - pos_durations[idx][2].item() < MIN_INTERVAL_SIZE:
            raise Exception("Positive interval size is less than the minimum interval size")

    neg_pairs, neg_intervals, neg_labels = [], [], []
    for idx in range(len(neg_durations)):
        neg_pairs.append((neg_durations[idx][0], neg_durations[idx][1]))
        neg_intervals.append((neg_durations[idx][2].item(), neg_durations[idx][3].item()))
        neg_labels.append(neg_durations[idx][4])

        if neg_durations[idx][3].item() - neg_durations[idx][2].item() < MIN_INTERVAL_SIZE:
            raise Exception("Negative interval size is less than the minimum interval size")

    samples = {'zero': {'pairs': zero_pairs, 'intervals': zero_intervals, 'labels': zero_labels},
               'pos': {'pairs': pos_pairs, 'intervals': pos_intervals, 'labels': pos_labels},
               'neg': {'pairs': neg_pairs, 'intervals': neg_intervals, 'labels': neg_labels}}

    return samples

def get_durations(data_dict, directed, nodes_num, init_time, last_time):
    """
    Get the durations from the data dictionary
    :param data_dict: the data dictionary
    :param directed: whether the graph is directed
    :param nodes_num: number of nodes
    :param init_time: initial time
    :param last_time: last time
    """

    pos_durations, neg_durations, zero_durations = [], [], []
    for i in range(nodes_num):

        for j in range(0 if directed else i+1, nodes_num):

            if i != j:

                # If the pair (i, j) contains any event
                if i in data_dict and j in data_dict[i]:

                    # Sort the time-state list by time
                    time_state_list = sorted(data_dict[i][j], key=lambda x: x[0])

                    for idx, (time, state) in enumerate(time_state_list):

                        # For the first event time
                        if idx == 0:

                            # If the time is greater than the initial time, than add the duration [init_time, time]
                            if time > init_time and time-init_time > MIN_INTERVAL_SIZE:

                                # Assumption: the first event time must indicate a positive or negative link
                                duration = (i, j, init_time, time, 0)
                                zero_durations.append(duration)

                        else:

                            # For duration [time_state_list[idx-1][0], time_state_list[idx][0]]
                            # The state of the duration is the state of the previous event
                            if time > time_state_list[idx-1][0] and time-time_state_list[idx-1][0] > MIN_INTERVAL_SIZE:
                                duration = (i, j, time_state_list[idx-1][0], time, time_state_list[idx-1][1])

                                if duration[4] == 1:
                                    pos_durations.append(duration)
                                elif duration[4] == -1:
                                    neg_durations.append(duration)
                                else:
                                    zero_durations.append(duration)

                    # If the last event time is smaller than the last time of the network
                    # Add the duration [time_state_list[-1][0], last_time]
                    if last_time != time_state_list[-1][0] and last_time-time_state_list[-1][0] > MIN_INTERVAL_SIZE:

                        duration = (i, j, time_state_list[-1][0], last_time, time_state_list[-1][1])
                        if duration[4] == 1:
                            pos_durations.append(duration)
                        elif duration[4] == -1:
                            neg_durations.append(duration)
                        else:
                            zero_durations.append(duration)

                # If the pair (i, j) does not contain any event
                else:

                    duration = (i, j, init_time, last_time, 0)
                    zero_durations.append(duration)

    return pos_durations, zero_durations, neg_durations

########################################################################################################################
# Construct the samples for the residual network
# Read the residual dataset
residual_dataset = Dataset()
residual_dataset.read_edge_list(os.path.join(split_folder, "./residual.edges"))
residual_data_dict = residual_dataset.get_data_dict(weights=True)

res_pos_dur, res_zero_dur, res_neg_dur = get_durations(residual_data_dict, is_directed, nodes_num, init_time, last_time)
res_samples = durations2samples(res_pos_dur, res_zero_dur, res_neg_dur, max_sample_size)

# Construct the samples for the completion set
# Read the residual dataset
comp_dataset = Dataset()
comp_dataset.read_edge_list(os.path.join(split_folder, "./completion.edges"))
comp_data_dict = comp_dataset.get_data_dict(weights=True)

comp_pos_dur, comp_zero_dur, comp_neg_dur = get_durations(comp_data_dict, is_directed, nodes_num, init_time, last_time)
comp_samples = durations2samples(comp_pos_dur, comp_zero_dur, comp_neg_dur, max_sample_size)

# Construct the samples for the prediction set
# Read the residual dataset
pred_dataset = Dataset()
pred_dataset.read_edge_list(os.path.join(split_folder, "./prediction.edges"))
pred_data_dict = pred_dataset.get_data_dict(weights=True)

pred_pos_dur, pred_zero_dur, pred_neg_dur = get_durations(
    pred_data_dict, is_directed, nodes_num, init_time=last_time, last_time=pred_dataset.get_last_time()
)
pred_samples = durations2samples(pred_pos_dur, pred_zero_dur, pred_neg_dur, max_sample_size)

# ######################################################################################################################
# Save the samples

# If the samples folder does not exist, create it
if not os.path.exists(samples_folder):
    os.makedirs(samples_folder)

with open(os.path.join(samples_folder, "reconstruction.samples"), 'wb') as f:
    pickle.dump(res_samples, f)

# Save the completion samples
with open(os.path.join(samples_folder, "completion.samples"), 'wb') as f:
    pickle.dump(comp_samples, f)

# Save the prediction samples
with open(os.path.join(samples_folder, "prediction.samples"), 'wb') as f:
    pickle.dump(pred_samples, f)

# ######################################################################################################################

