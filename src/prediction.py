import matplotlib.pyplot as plt
import torch
from src.base import BaseModel
import utils
from utils import prior
import os

class PredictionModel(BaseModel, torch.nn.Module):

    def __init__(self, x0_s: torch.Tensor, v_s: torch.Tensor, beta_s: torch.Tensor, 
                directed = True, init_states: torch.BoolTensor = None, prior_lambda: float = 1.0, 
                prior_sigma_s: float = 1.0, prior_B_x0_logit_c_s: float = 1.0, 
                prior_B_ls_s: float = 1.0, prior_C_Q_s: torch.Tensor = None, prior_R_factor_inv_s: torch.Tensor = None, 
                x0_r: torch.Tensor = None, v_r: torch.Tensor = None, beta_r: torch.Tensor = None, 
                prior_sigma_r: float = 1.0, prior_B_x0_logit_c_r: float = 1.0, 
                prior_B_ls_r: float = 1.0, prior_C_Q_r: torch.Tensor = None, prior_R_factor_inv_r: torch.Tensor = None,
                device: torch.device = "cpu", verbose: bool = False, seed: int = 19):

        super(PredictionModel, self).__init__(
                x0_s=x0_s, v_s=v_s, beta_s=beta_s, 
                directed = directed, init_states =init_states, prior_lambda = prior_lambda, 
                prior_sigma_s=prior_sigma_s, prior_B_x0_logit_c_s=prior_B_x0_logit_c_r, 
                prior_B_ls_s=prior_B_ls_s, prior_C_Q_s=prior_C_Q_s, prior_R_factor_inv_s=prior_R_factor_inv_s, 
                x0_r=x0_r, v_r=v_r, beta_r=beta_r, 
                prior_sigma_r=prior_sigma_r, prior_B_x0_logit_c_r=prior_B_x0_logit_c_r, 
                prior_B_ls_r=prior_B_ls_r, prior_C_Q_r=prior_C_Q_r, prior_R_factor_inv_r=prior_R_factor_inv_r,
                device=device, verbose=verbose, seed=seed
        )


    def get_inv_train_cov(self, nodes: torch.Tensor):

        # Define the final dimension size
        final_dim = self.get_nodes_num() * (self.get_bins_num()+1) * self.get_dim()

        # Covariance scaling coefficient
        lambda_sq = self.get_prior_lambda()**2

        # Noise for the Gaussian process
        sigma_sq_inv_s = 1.0 / (self.get_prior_sigma_s()**2)
        if self.is_directed():
            sigma_sq_inv_r = 1.0 / (self.get_prior_sigma_r()**2)

        # Get the factor of the covariance and the inverse factor of the capacitance matrices
        Kf_s, R_factor_inv_s, Kf_r, R_factor_inv_r = self.get_cov_factors(nodes=nodes, compute_R_factor_inv=True)

        # Normalize and vectorize the initial position and velocity vectors
        x0_s = torch.index_select(self.get_x0_s(), dim=0, index=nodes).flatten() 
        v_s = utils.vectorize(torch.index_select(self.get_v_s(),  dim=1, index=nodes)).flatten()
        x0v_s = torch.hstack((x0_s, v_s))
        if self.is_directed():
            x0_r = torch.index_select(self.get_x0_r(), dim=0, index=nodes).flatten() 
            v_r = utils.vectorize(torch.index_select(self.get_v_r(),  dim=1, index=nodes)).flatten()
            x0v_r = torch.hstack((x0_r, v_r))

        # Computation of the squared Mahalanobis distance: v.T @ inv(D + W @ W.T) @ v
        # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        mahalanobis_term1_s = torch.diag(sigma_sq_inv_s * torch.ones(size=(final_dim, ), dtype=torch.float, device=self.get_device()))
        f_s = Kf_s @ R_factor_inv_s.T
        mahalanobis_term2_s = f_s @ f_s.T
        m_s = (1.0 / lambda_sq) * ( mahalanobis_term1_s - mahalanobis_term2_s )
        if self.is_directed():
            mahalanobis_term1_r = torch.diag(sigma_sq_inv_r * torch.ones(size=(final_dim, ), dtype=torch.float, device=self.get_device()))
            f_r = Kf_r @ R_factor_inv_r.T
            mahalanobis_term2_r = f_r @ f_r.T
            m_r = (1.0 / lambda_sq) * ( mahalanobis_term1_r - mahalanobis_term2_r )
        else:
            m_r = None

        return m_s, m_r

    def get_test_train_cov(self, time_list: torch.Tensor):

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

        # B x B matrix
        B_s = prior.get_B(
            bin_centers1=time_list.view(1, len(time_list)), bin_centers2=middle_bounds, 
            prior_B_x0_c=self.get_prior_B_x0_c_s(), prior_B_ls=self.get_prior_B_ls_s()
        )
        # Remove the first column corresponding to initial position vectors
        B_s = B_s[:, 1:] 
        # N x K matrix where K is the community size
        C_factor_s = prior.get_C_factor(prior_C_Q=self.get_prior_C_Q_s())
        # D x D matrix
        D_factor_s = prior.get_D_factor(dim=self.get_dim())

        test_train_cov_s = self.get_prior_lambda()**2 * torch.kron(B_s @ torch.kron( (C_factor_s @ C_factor_s.T), (D_factor_s @ D_factor_s.T) ) )

        if self.is_directed():
            # B x B matrix
            B_r = prior.get_B(
                bin_centers1=time_list.view(1, len(time_list)), bin_centers2=middle_bounds, 
                prior_B_x0_c=self.get_prior_B_x0_c_r(), prior_B_ls=self.get_prior_B_ls_r()
            )
            # Remove the first column corresponding to initial position vectors
            B_r = B_r[:, 1:] 
            # N x K matrix where K is the community size
            C_factor_r = prior.get_C_factor(prior_C_Q=self.get_prior_C_Q_r())
            # D x D matrix
            D_factor_r = prior.get_D_factor(dim=self.get_dim())

            test_train_cov_r = self.get_prior_lambda()**2 * torch.kron(B_r @ torch.kron( (C_factor_r @ C_factor_r.T), (D_factor_r @ D_factor_r.T) ) )

        else:

            test_train_cov_r = None

        return test_train_cov_s, test_train_cov_r



    def get_v_pred(self, times_list: torch.Tensor):
        

        

        # # Get the bin bounds
        # bounds = self.get_bins_bounds()

        # # Get the middle time points of the bins for TxT covariance matrix
        # middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

        # # (B+1) x len(times_list) matrix
        # B = self.get_B_factor(
        #     bin_centers1=times_list.view(1, len(times_list)), bin_centers2=middle_bounds,
        #     prior_B_x0_c=self.get_prior_B_x0_c(), prior_B_ls=self.get_prior_B_ls(), only_kernel=True
        # )
        # B = B[:, 1:]  # remove the first row corresponding to initial position vectors

        # # N x K matrix where K is the community size
        # C_factor = self.get_C_factor(prior_C_Q=self.get_prior_C_Q())
        # # D x D matrix
        # D_factor = self.get_D_factor(dim=self.get_dim())

        # # Construct the test_train covariance matrix
        # test_train_cov = (self.get_prior_lambda() ** 2) * torch.kron(
        #     B, torch.kron((C_factor @ C_factor.T), (D_factor @ D_factor.T))
        # )

        # # Normalize and vectorize the initial position and velocity vectors
        # x0 = vectorize(self.get_x0())
        # v_batch = vectorize(self.get_v()).flatten()

        # # Compute the estimated velocity vectors
        # x0v = torch.hstack((x0, v_batch))
        # est = unvectorize(
        #     test_train_cov.T @ self.get_inv_train_cov() @ x0v,
        #     size=(len(times_list), self.get_number_of_nodes(), self.get_dim())
        # )

        # # Add noise
        # est = est + torch.normal(mean=torch.zeros(size=(len(times_list), self.get_number_of_nodes(), self.get_dim())), std=self.get_prior_sigma())

        return est

    def get_x_pred(self, times_list: torch.Tensor):

        rlast_s = self.get_rt_s(
            time_list=torch.ones(self.get_nodes_num(), dtype=torch.float, device=self.get_device())*self.get_last_time(), 
            nodes=torch.arange(self.get_nodes_num(), dtype=torch.int, device=self.get_device())
        )
        if self.is_directed():
            rlast_t = self.get_rt_s(
                time_list=torch.ones(self.get_nodes_num(), dtype=torch.float, device=self.get_device())*self.get_last_time(), 
                nodes=torch.arange(self.get_nodes_num(), dtype=torch.int, device=self.get_device())
            )

        ##########################
        ######################

        # A matrix of size N x D
        x_last = self.get_xt(
            events_times_list=torch.as_tensor([self.get_last_time()]*self.get_number_of_nodes()),
            x0=self.get_x0(), v=self.get_v()
        )

        # Get the estimated velocity matrix of size len(time_samples) x N x D
        pred_v = self.get_v_pred(times_list=times_list)

        # A matrix of size len(time_samples) x N x D
        pred_x = x_last.unsqueeze(0) + (times_list - self.get_last_time()).view(-1, 1, 1) * pred_v
        return pred_x


# Define the model name
model_name = f"synthetic_march14_directed_B=100_D=2_lr=0.01_ep=30_lambda=10000.0"

# # Get the current folder path
current_folder = os.path.dirname(os.path.abspath(__file__))
# Define the model path
model_path = os.path.join(os.path.join(current_folder, '..', 'models'), model_name + ".model")
kwargs, lm_state =  torch.load(model_path)

lm_state = {key.replace('_BaseModel__', ''): value for key, value in lm_state.items()}

# print({**kwargs, **lm_state}.keys())
pm = PredictionModel(**{**kwargs, **lm_state})
state = {'x0_s': 0, 'v_s':1, 'beta_s': 2}
# pm = PredictionModel(**state)
# pm.load_state_dict(lm_state)

print("---")
# print(model_state_dict['model_state_dict'].keys())

# class PredictionModel(BaseModel, torch.nn.Module):

#     def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, bins_num: int = 100,
#                  last_time: float = 1.0, prior_lambda: float = 1e5, prior_sigma: torch.Tensor = None,
#                  prior_B_x0_c: torch.Tensor = None, prior_B_ls: torch.Tensor = None, prior_C_Q: torch.Tensor= None,
#                  device: torch.device = "cpu", verbose: bool = False, seed: int = 0, **kwargs):

#         kwargs = kwargs

#         # super(LearningModel, self).__init__(
#         #     x0=kwargs["x0"],
#         #     v=torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
#         #     beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num,), device=device) - 1, requires_grad=False),
#         #     bins_num=bins_num,
#         #     last_time=last_time,
#         #     prior_lambda=prior_lambda,
#         #     prior_sigma=torch.nn.Parameter(
#         #         (2.0 / bins_num) * torch.rand(size=(1,), device=device) + (1. / bins_num), requires_grad=False
#         #     ),
#         #     prior_B_x0_c=torch.nn.Parameter(torch.ones(size=(1, 1), device=device), requires_grad=False),
#         #     prior_B_sigma=torch.nn.Parameter(
#         #         (1 - (2.0 / bins_num)) * torch.rand(size=(1,), device=device) + (1. / bins_num), requires_grad=False
#         #     ),
#         #     prior_C_Q=torch.nn.Parameter(torch.rand(size=(nodes_num, prior_k), device=device), requires_grad=False),
#         #     device=device,
#         #     verbose=verbose,
#         #     seed=seed
#         # )

#         # super(PredictionModel, self).__init__(
#         #     **{key.replace('_BaseModel__', ''): value for key, value in kwargs.items()}
#         # )
#         super(PredictionModel, self).__init__(
#             x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=last_time, prior_lambda=prior_lambda,
#             prior_sigma=prior_sigma, prior_B_x0_c=prior_B_x0_c, prior_B_ls=prior_B_ls, prior_C_Q=prior_C_Q,
#             device=device, verbose=verbose, seed=seed, kwargs=kwargs
#         )

#         self.__inv_train_cov = self.__compute_inv_train_cov()

#     def get_inv_train_cov(self):

#         return self.__inv_train_cov

#     def __compute_inv_train_cov(self):

#         # Some scalars
#         sigma_sq = torch.clamp(self.get_prior_sigma(), min=5. / self.get_bins_num()) ** 2
#         sigma_sq_inv = 1.0 / sigma_sq
#         lambda_sq = self.get_prior_lambda() ** 2
#         reduced_dim = self.get_prior_k() * (self.get_bins_num() + 1) * self.get_dim()
#         final_dim = self.get_number_of_nodes() * (self.get_bins_num() + 1) * self.get_dim()

#         # Get the bin bounds
#         bounds = self.get_bins_bounds()

#         # Get the middle time points of the bins for TxT covariance matrix
#         middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

#         # B x B matrix
#         B_factor = self.get_B_factor(
#             bin_centers1=middle_bounds, bin_centers2=middle_bounds,
#             prior_B_x0_c=self.get_prior_B_x0_c(), prior_B_ls=self.get_prior_B_ls()
#         )
#         # N x K matrix where K is the community size
#         C_factor = self.get_C_factor(prior_C_Q=self.get_prior_C_Q())
#         # D x D matrix
#         D_factor = self.get_D_factor(dim=self.get_dim())

#         # Compute the capacitance matrix R only if batch_num == 0
#         R = torch.eye(reduced_dim) + sigma_sq_inv * torch.kron(
#             B_factor.T @ B_factor, torch.kron(C_factor.T @ C_factor, D_factor.T @ D_factor)
#         )
#         R_factor = torch.linalg.cholesky(R) #R_inv = torch.inverse(R + (10*EPS) * torch.eye(n=R.shape[0], m=R.shape[1])) #R_factor = torch.linalg.cholesky(R)
#         R_factor_inv = torch.inverse(R_factor).T #R_factor_inv = torch.linalg.cholesky(R_inv, upper=False) #R_factor_inv = torch.inverse(R_factor)

#         # Computation of inv(D + W @ W.T)
#         # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
#         # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
#         term1 = torch.diag(sigma_sq_inv*torch.ones(size=(final_dim, ), dtype=torch.float, device=self.get_device()))
#         K_factor = torch.kron(B_factor.contiguous(), torch.kron(C_factor, D_factor))
#         f = (sigma_sq_inv * K_factor @ R_factor_inv) #(sigma_sq_inv * K_factor @ R_factor_inv.T)
#         term2 = f @ f.T

#         return (1.0 / lambda_sq) * (term1 - term2)

#     def get_v_pred(self, times_list: torch.Tensor):

#         # Get the bin bounds
#         bounds = self.get_bins_bounds()

#         # Get the middle time points of the bins for TxT covariance matrix
#         middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

#         # (B+1) x len(times_list) matrix
#         B = self.get_B_factor(
#             bin_centers1=times_list.view(1, len(times_list)), bin_centers2=middle_bounds,
#             prior_B_x0_c=self.get_prior_B_x0_c(), prior_B_ls=self.get_prior_B_ls(), only_kernel=True
#         )
#         B = B[:, 1:]  # remove the first row corresponding to initial position vectors

#         # N x K matrix where K is the community size
#         C_factor = self.get_C_factor(prior_C_Q=self.get_prior_C_Q())
#         # D x D matrix
#         D_factor = self.get_D_factor(dim=self.get_dim())

#         # Construct the test_train covariance matrix
#         test_train_cov = (self.get_prior_lambda() ** 2) * torch.kron(
#             B, torch.kron((C_factor @ C_factor.T), (D_factor @ D_factor.T))
#         )

#         # Normalize and vectorize the initial position and velocity vectors
#         x0 = vectorize(self.get_x0())
#         v_batch = vectorize(self.get_v()).flatten()

#         # Compute the estimated velocity vectors
#         x0v = torch.hstack((x0, v_batch))
#         est = unvectorize(
#             test_train_cov.T @ self.get_inv_train_cov() @ x0v,
#             size=(len(times_list), self.get_number_of_nodes(), self.get_dim())
#         )

#         # Add noise
#         est = est + torch.normal(mean=torch.zeros(size=(len(times_list), self.get_number_of_nodes(), self.get_dim())), std=self.get_prior_sigma())

#         return est

#     def get_x_pred(self, times_list: torch.Tensor):

#         # A matrix of size N x D
#         x_last = self.get_xt(
#             events_times_list=torch.as_tensor([self.get_last_time()]*self.get_number_of_nodes()),
#             x0=self.get_x0(), v=self.get_v()
#         )

#         # Get the estimated velocity matrix of size len(time_samples) x N x D
#         pred_v = self.get_v_pred(times_list=times_list)

#         # A matrix of size len(time_samples) x N x D
#         pred_x = x_last.unsqueeze(0) + (times_list - self.get_last_time()).view(-1, 1, 1) * pred_v
#         return pred_x

#     def get_intensity_integral_pred(self, t_init: float, t_last: float, sample_size=100):

#         assert t_init >= self.get_last_time(), \
#             "The given boundary times must be larger than the last time of the training data!"

#         time_samples = torch.linspace(t_init, t_last, steps=sample_size)[:-1]  # Discard the last time point
#         delta_t = time_samples[1] - time_samples[0]

#         # N x N
#         beta_mat = self.get_beta().unsqueeze(1) + self.get_beta().unsqueeze(0)
#         # (sample_size-1) x N x D
#         xt = self.get_x_pred(times_list=time_samples)
#         # (sample_size-1) x (N x N)
#         delta_x_mat = torch.cdist(xt, xt, p=2)
#         # (sample_size-1) x (N x N)
#         lambda_pred = torch.sum(torch.exp(beta_mat.unsqueeze(0) - delta_x_mat), dim=0) * delta_t

#         return lambda_pred

