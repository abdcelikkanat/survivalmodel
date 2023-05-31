import os
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.common import set_seed, flatIdx2matIdx, matIdx2flatIdx
import numpy as np
from src.dataset import Dataset
import pickle


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
dataset.read_edgelist(split_folder + "/first_half.edges")
dataset.print_info()
init_time, last_time = dataset.get_init_time(), dataset.get_last_time()
nodes_num = dataset.get_nodes_num()
is_directed = dataset.is_directed()
is_signed = dataset.is_signed()
del dataset

########################################################################################################################


def durations2samples(pos_durations, zero_durations, neg_durations, max_sample_size):

    # Construct the zero samples
    zero_chosen_idx = np.random.choice(
        len(zero_durations), size=(min(max_sample_size, len(zero_durations), len(pos_durations) + len(neg_durations)),),
        replace=False
    )
    zero_pairs, zero_times, zero_labels = [], [], []
    for idx in zero_chosen_idx:
        zero_pairs.append((zero_durations[idx][0].item(), zero_durations[idx][1].item()))
        zero_times.append((zero_durations[idx][2].item() + zero_durations[idx][3].item())/2)
        zero_labels.append(zero_durations[idx][4].item())

    pos_pairs, pos_times, pos_labels = [], [], []
    for idx in range(len(pos_durations)):
        pos_pairs.append((pos_durations[idx][0].item(), pos_durations[idx][1].item()))
        pos_times.append((pos_durations[idx][2].item() + pos_durations[idx][3].item())/2)
        pos_labels.append(pos_durations[idx][4].item())

    neg_pairs, neg_times, neg_labels = [], [], []
    for idx in range(len(neg_durations)):
        neg_pairs.append((neg_durations[idx][0].item(), neg_durations[idx][1].item()))
        neg_times.append((neg_durations[idx][2].item() + neg_durations[idx][3].item())/2)
        neg_labels.append(neg_durations[idx][4].item())

    samples = {'zero': {'pairs': zero_pairs, 'times': zero_times, 'labels': zero_labels},
               'pos': {'pairs': pos_pairs, 'times': pos_times, 'labels': pos_labels},
               'neg': {'pairs': neg_pairs, 'times': neg_times, 'labels': neg_labels}}

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
                            if time > init_time:

                                # Assumption: the first event time must indicate a positive or negative link
                                duration = (i, j, init_time, time, 0)
                                zero_durations.append(duration)

                        else:

                            # For duration [time_state_list[idx-1][0], time_state_list[idx][0]]
                            # The state of the duration is the state of the previous event
                            duration = (i, j, time_state_list[idx-1][0], time, time_state_list[idx-1][1])

                            if duration[4] == 1:
                                pos_durations.append(duration)
                            elif duration[4] == -1:
                                neg_durations.append(duration)
                            else:
                                zero_durations.append(duration)

                    # If the last event time is smaller than the last time of the network
                    # Add the duration [time_state_list[-1][0], last_time]
                    if last_time != time_state_list[-1][0]:

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
residual_dataset.read_edgelist(os.path.join(split_folder, "./residual.edges"))
residual_data_dict = residual_dataset.get_data_dict()

res_pos_dur, res_zero_dur, res_neg_dur = get_durations(residual_data_dict, is_directed, nodes_num, init_time, last_time)
res_samples = durations2samples(res_pos_dur, res_zero_dur, res_neg_dur, max_sample_size)

# Construct the samples for the completion set
# Read the residual dataset
comp_dataset = Dataset()
comp_dataset.read_edgelist(os.path.join(split_folder, "./completion.edges"))
comp_data_dict = comp_dataset.get_data_dict()

comp_pos_dur, comp_zero_dur, comp_neg_dur = get_durations(comp_data_dict, is_directed, nodes_num, init_time, last_time)
comp_samples = durations2samples(comp_pos_dur, comp_zero_dur, comp_neg_dur, max_sample_size)

# Construct the samples for the prediction set
# Read the residual dataset
pred_dataset = Dataset()
pred_dataset.read_edgelist(os.path.join(split_folder, "./prediction.edges"))
pred_data_dict = pred_dataset.get_data_dict()

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

