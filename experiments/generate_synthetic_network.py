import os
from src.construction import ConstructionModel


# Define the dataset name
dataset_name = f"synthetic_crazy2"

# Define the model parameters
args = {
    'cluster_sizes': [8]*10, 'bins_num': 100, 'dim': 2, 'directed': False,
    'beta_s': [3, 0.], 'beta_r': [-0.1, 0.], #[1., -1.], 'beta_r': [-0.1, 0.],
    'prior_lambda': 3e1,
    'prior_sigma_s': 1e-2, 'prior_sigma_r': 7.5e-2,
    'prior_B_x0_s': 1., 'prior_B_x0_r': 1e6,
    'prior_B_ls_s': 1e-6, 'prior_B_ls_r': 0.5e-1, #0.5e-1, 'prior_B_ls_r': 0.5e-1,
    'device': "cpu", 'verbose': True, 'seed': 16,
}
# args = {
#     'cluster_sizes': [15]*10, 'bins_num': 100, 'dim': 2, 'directed': False,
#     'beta_s': [-2., -5.], 'beta_r': [-0.1, 0.], #[1., -1.], 'beta_r': [-0.1, 0.],
#     'prior_lambda': 3.0,
#     'prior_sigma_s': 7.5e-2, 'prior_sigma_r': 7.5e-2,
#     'prior_B_x0_logit_c_s': 1e6, 'prior_B_x0_logit_c_r': 1e6,
#     'prior_B_ls_s': 1e+0, 'prior_B_ls_r': 0.5e-1, #0.5e-1, 'prior_B_ls_r': 0.5e-1,
#     'device': "cpu", 'verbose': True, 'seed': 16,
# }
# Initialize the constructor model
cm = ConstructionModel(**args)

# Define the dataset folder path
dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', dataset_name)

# If the dataset folder does not exist, create it
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Save the generated dataset and model
cm.save_model(file_path=os.path.join(dataset_folder, f"{dataset_name}.model"))
cm.write_edges(file_path=os.path.join(dataset_folder, f"{dataset_name}.edges"))
cm.save_animation(file_path=os.path.join(dataset_folder, f"{dataset_name}.mp4"))
cm.write_info(file_path=os.path.join(dataset_folder, f"{dataset_name}.info"))
