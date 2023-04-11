import os
import torch
import utils
import pickle as pkl
from src.sdp import SurviveDieProcess
from src.base import BaseModel
from utils import prior


class ConstructionModel(torch.nn.Module):

    def __init__(self, cluster_sizes: list, bins_num: int, dim: int, directed: bool,
                prior_lambda: float, prior_sigma_s: float, prior_sigma_r: float, 
                prior_B_x0_logit_c_s: float, prior_B_x0_logit_c_r: float, prior_B_ls_s: float, prior_B_ls_r: float, 
                device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        super(ConstructionModel, self).__init__()

        # Set the model parameters
        self.__nodes_num = sum(cluster_sizes)
        self.__cluster_sizes = cluster_sizes
        self.__k = len(cluster_sizes)
        self.__dim = dim
        self.__bins_num = bins_num
        self.__directed = directed

        # Set the prior hyperparameters
        self.__prior_lambda = torch.as_tensor(prior_lambda, dtype=torch.float, device=device)
        self.__prior_sigma_s = torch.as_tensor(prior_sigma_s, dtype=torch.float, device=device)
        self.__prior_sigma_r = torch.as_tensor(prior_sigma_r, dtype=torch.float, device=device) if self.__directed else None
        self.__prior_B_x0_logit_c_s = torch.as_tensor(prior_B_x0_logit_c_s, dtype=torch.float, device=device)
        self.__prior_B_x0_logit_c_r = torch.as_tensor(prior_B_x0_logit_c_r, dtype=torch.float, device=device) if self.__directed else None
        self.__prior_B_ls_s = torch.as_tensor(prior_B_ls_s, dtype=torch.float, device=device)
        self.__prior_B_ls_r = torch.as_tensor(prior_B_ls_r, dtype=torch.float, device=device) if self.__directed else None

        # Construct the Q matrix for the node matrix C, (nodes)
        self.__prior_C_Q_s = -utils.INF * torch.ones(size=(self.__nodes_num, self.__k), dtype=torch.float)
        for k in range(self.__k):
            self.__prior_C_Q_s[sum(self.__cluster_sizes[:k]):sum(self.__cluster_sizes[:k + 1]), k] = 0
        self.__prior_C_Q_r = self.__prior_C_Q_s if self.__directed else None
        
        #
        self.__device = device
        self.__verbose = verbose
        self.__seed = seed

        #
        self.__bm = None
        self.__data = None 
        
        self.__bm, self.__data = self.sample_graph()

    def sample_graph(self):
        '''
        Sample a new graph
        '''

        # Define the number of pairs that can have an event
        pairs_num = self.__nodes_num*(self.__nodes_num-1)  if self.__directed else self.__nodes_num*(self.__nodes_num-1)//2

        # Constuct the node pairs
        node_pairs = torch.triu_indices(self.__nodes_num, self.__nodes_num, offset=1, device=self.__device)
        # If directed, add the lower triangular part
        if self.__directed:
            node_pairs = torch.cat((
                node_pairs, torch.tril_indices(self.__nodes_num, self.__nodes_num, offset=-1, device=self.__device
            )), dim=1)
        
        # Set the initial states
        init_states = torch.zeros(size=(pairs_num, ), device=self.__device, dtype=torch.int)
        
        # Sample the initial position and velocity tensors
        x0_s, v_s, x0_r, v_r = self.sample_x0_and_v()

        # Sample the bias tensors
        beta_s = 0*torch.randn(size=(self.__nodes_num, 2), dtype=torch.float, device=self.__device,)
        beta_r = 0*torch.randn(size=(self.__nodes_num, 2), dtype=torch.float, device=self.__device) if self.__directed else None

        bm = BaseModel(
            x0_s = x0_s, v_s = v_s, beta_s = beta_s, 
            init_states=init_states, directed = self.__directed, prior_lambda = self.__prior_lambda,
            prior_sigma_s = self.__prior_sigma_s, prior_B_x0_logit_c_s = self.__prior_B_x0_logit_c_s, 
            prior_B_ls_s = self.__prior_B_ls_s, prior_C_Q_s = self.__prior_C_Q_s, prior_R_factor_inv_s = None,
            x0_r = x0_r, v_r = v_r, beta_r = beta_r,
            prior_sigma_r = self.__prior_sigma_r, prior_B_x0_logit_c_r = self.__prior_B_x0_logit_c_r, 
            prior_B_ls_r = self.__prior_B_ls_r, prior_C_Q_r = self.__prior_C_Q_r,  prior_R_factor_inv_r = None,
            device=self.__device, verbose = self.__verbose, seed=self.__seed
        )

        pair_events, pair_states = self.__sample_events(bm=bm, node_pairs=node_pairs)

        tiplets = [(tuple(pair), event, state) for pair in node_pairs.T.tolist() for event, state in zip(pair_events[(pair[0], pair[1])], pair_states[(pair[0], pair[1])])]
        pairs, events, states = zip(*sorted(tiplets, key=lambda tri: tri[1]))
        data = list(pairs), torch.as_tensor(events, dtype=torch.float), torch.as_tensor(states, dtype=torch.int8), self.__directed, 0, 1.0

        return bm, data

    def sample_x0_and_v(self):
        '''
        Sample the initial position and velocity of the nodes
        For directed graphs, B matrix (bins) are only different

        '''

        # Define the bin centers
        bin_centers = torch.arange(0.5/self.__bins_num, 1.0, 1.0/self.__bins_num).unsqueeze(0)

        # Define the final dimension size
        final_dim = self.__nodes_num * (self.__bins_num+1) * self.__dim
        
        # Construct the factor of B matrix (bins)
        B_factor_s = prior.get_B_factor(bin_centers, bin_centers, self.__prior_B_x0_logit_c_s, self.__prior_B_ls_s)
        # Construct the factor of C matrix (nodes)
        C_factor_s = prior.get_C_factor(self.__prior_C_Q_s)
        # Get the factor of D matrix, (dimension)
        D_factor_s = prior.get_D_factor(dim=self.__dim)

        # Sample from the low rank multivariate normal distribution
        # covariance_matrix = cov_factor @ cov_factor.T + cov_diag
        cov_factor_s = self.__prior_lambda * torch.kron(B_factor_s.contiguous(), torch.kron(C_factor_s, D_factor_s))
        cov_diag_s = (self.__prior_lambda**2) * (self.__prior_sigma_s**2) * torch.ones(size=(final_dim, ), dtype=torch.float, device=self.__device)
        # Sample from the low rank multivariate normal distribution
        sample_s = torch.distributions.LowRankMultivariateNormal(
            loc=torch.zeros(size=(final_dim,)),
            cov_factor=cov_factor_s,
            cov_diag=cov_diag_s
        ).sample().reshape(shape=(self.__bins_num+1, self.__nodes_num, self.__dim))

        # Split the tensor into x0 and v
        x0_s, v_s = torch.split(sample_s, [1, self.__bins_num])
        x0_s = x0_s.squeeze(0)
        
        if self.__directed:

            # Construct the factor of B matrix (bins)
            B_factor_r = prior.get_B_factor(bin_centers, bin_centers, self.__prior_B_x0_logit_c_r, self.__prior_B_ls_r)
            # Construct the factor of C matrix (nodes)
            C_factor_r = prior.get_C_factor(self.__prior_C_Q_r)
            # Get the factor of D matrix, (dimension)
            D_factor_r = prior.get_D_factor(dim=self.__dim)

            # Sample from the low rank multivariate normal distribution
            # covariance_matrix = cov_factor @ cov_factor.T + cov_diag
            cov_factor_r = self.__prior_lambda * torch.kron(B_factor_r.contiguous(), torch.kron(C_factor_r, D_factor_r))
            cov_diag_r = (self.__prior_lambda**2) * (self.__prior_sigma_r**2) * torch.ones(size=(final_dim, ), dtype=torch.float, device=self.__device)

            # Sample from the low rank multivariate normal distribution
            sample_r = torch.distributions.LowRankMultivariateNormal(
                loc=torch.zeros(size=(final_dim,)),
                cov_factor=cov_factor_r,
                cov_diag=cov_diag_r
            ).sample().reshape(shape=(self.__bins_num+1, self.__nodes_num, self.__dim))

            # Split the tensor into x0 and v
            x0_r, v_r = torch.split(sample_r, [1, self.__bins_num])
            x0_r = x0_r.squeeze(0)
        
        else:
            x0_r, v_r = None, None


        return utils.standardize(x0_s), utils.standardize(v_s), utils.standardize(x0_r), utils.standardize(v_r)


    def __sample_events(self, node_pairs: torch.Tensor, bm: BaseModel) -> dict:

        # Construct a dictionary of dictionaries to store the events times
        pair_events = {tuple(pair): [] for pair in node_pairs.T.tolist()}
        pair_states = {tuple(pair): [] for pair in node_pairs.T.tolist()}

        # Get the positions at the beginning of each time bin for every node
        rt_s = bm.get_rt_s(
            time_list=bm.get_bin_bounds()[:-1].repeat(self.__nodes_num), 
            nodes=torch.repeat_interleave(torch.arange(self.__nodes_num), repeats=self.__bins_num)
        )
        rt_s = rt_s.reshape((self.__nodes_num, self.__bins_num,  self.__dim)).transpose(0, 1)
        if self.__directed:
            rt_r = bm.get_rt_r(
                time_list=bm.get_bin_bounds()[:-1].repeat(self.__nodes_num), 
                nodes=torch.repeat_interleave(torch.arange(self.__nodes_num), repeats=self.__bins_num)
            ).reshape((self.__nodes_num, self.__bins_num,  self.__dim)).transpose(0, 1)
        else:
            rt_r = rt_s

        for pair in node_pairs.T:

            i, j = pair.tolist()

            # Define the intensity function for each node pair (i,j)
            intensity_func = lambda t, state: bm.get_intensity_at(
                time_list=torch.as_tensor([t]), pairs=pair.unsqueeze(1), states=torch.as_tensor([state]).type(torch.long)
            ).item()

            # Get the flat index of the pair
            flat_idx = utils.pairIdx2flatIdx(i, j, self.__nodes_num, directed=self.__directed)

            # Get the critical points
            v_s = bm.get_v_s()
            v_r = bm.get_v_r() if self.__directed else v_s
            critical_points = self.__get_critical_points(i=i, j=j, bin_bounds=bm.get_bin_bounds(), rt_s=rt_s, rt_r=rt_r, v_s=v_s, v_r=v_r)

            # Get the state
            initial_state = bm.get_init_states(flat_idx)

            # Simulate the Survive or Die Process
            nhpp_ij = SurviveDieProcess(
                lambda_func=intensity_func, initial_state=initial_state, critical_points=critical_points,
                seed=self.__seed + flat_idx
            )
            ij_edge_times, ij_edge_states = nhpp_ij.simulate()
            # Add the event times
            pair_events[(i, j)].extend(ij_edge_times)
            pair_states[(i, j)].extend(ij_edge_states)

        return pair_events, pair_states


    def __get_critical_points(self, i: int, j: int, bin_bounds: torch.Tensor, rt_s: torch.Tensor, rt_r: torch.Tensor, v_s: torch.Tensor, v_r: torch.Tensor) -> tuple:

        # Add the initial time point
        critical_points = []

        for idx in range(self.__bins_num):

            interval_init_time = bin_bounds[idx]
            interval_last_time = bin_bounds[idx+1]

            # Add the initial time point of the interval
            critical_points.append(interval_init_time)

            # Get the differences
            delta_idx_x = rt_s[idx, i, :] - rt_r[idx, j, :]
            delta_idx_v = v_s[idx, i, :] - v_r[idx, j, :] 

            # For the model containing only position and velocity
            # Find the point in which the derivative equal to 0
            t = - torch.dot(delta_idx_x, delta_idx_v) / (torch.dot(delta_idx_v, delta_idx_v) + utils.EPS) + interval_init_time

            if interval_init_time < t < interval_last_time:
                critical_points.append(t)

        # Add the last time point
        critical_points.append(bin_bounds[-1])

        return critical_points

    def get_data(self):

        return self.__data

    def get_model(self):

        return self.__bm

    def save_data(self, file_path: str):

        with open(file_path, 'wb') as f:
            pkl.dump(self.__data, f)

    def save_model(self, file_path: str):

        with open(file_path, 'wb') as f:
            pkl.dump(self.__bm, f)

    def write_edges(self, file_path: str):
        '''
        Write the edges
        '''
        with open(file_path, 'w') as f:
            for pair, event_time, state in zip(self.__data[0], self.__data[1], self.__data[2]):
                f.write(f"{pair[0]} {pair[1]} {event_time} {state}\n")