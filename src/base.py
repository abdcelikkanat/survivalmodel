import utils
import torch
import numpy as np
import random
from utils import prior

class BaseModel(torch.nn.Module):
    '''
    Description
    '''
    def __init__(self, x0: tuple, v: tuple, beta: tuple, 
                bins_num: int = 100, directed = True, init_states: torch.BoolTensor = None, 
                prior_lambda: float = 1.0, prior_sigma: float = 1.0, 
                prior_B_x0_c: float = 1.0, prior_B_ls: float = 1.0, 
                prior_C_Q: torch.Tensor = None, prior_R_factor_inv: torch.Tensor = None,
                device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        super(BaseModel, self).__init__()

        # Define the constants
        self.__init_time = 0.0
        self.__last_time = 1.0

        # Set the model parameters
        self.__x0_s = x0[0]
        self.__x0_r = x0[1] if directed else None
        self.__v_s = v[0]
        self.__v_r = v[1] if directed else None
        self.__beta_s = beta[0]
        self.__beta_r = beta[1] if directed else None

        # Set the initial states, number of bins and directed variable
        self.__init_states = init_states
        self.__bins_num = bins_num
        self.__directed = directed
        # Set the number of nodes, dimension size and bin width
        self.__nodes_num = self.__x0_s.shape[0]
        self.__dim = self.__x0_s.shape[1]
        self.__bin_width = torch.tensor((self.__last_time - self.__init_time) / float(self.__bins_num), device=device)

        # Set the model hypermaters
        self.__prior_lambda = prior_lambda
        self.__prior_sigma_s = prior_sigma[0]
        self.__prior_sigma_r = prior_sigma[1] if directed else None
        self.__prior_B_x0_c_s = prior_B_x0_c[0]
        self.__prior_B_x0_c_r = prior_B_x0_c[1] if directed else None
        self.__prior_B_ls_s = prior_B_ls[0]
        self.__prior_B_ls_r = prior_B_ls[1] if directed else None
        self.__prior_C_Q_s = prior_C_Q[0]
        self.__prior_C_Q_r = prior_C_Q[1] if directed else None
        self.__prior_R_factor_inv_s = prior_R_factor_inv[0]
        self.__prior_R_factor_inv_r = prior_R_factor_inv[1] if directed else None

        self.__device = device
        self.__verbose = verbose
        self.__seed = seed

        

        # Set the seed value for reproducibility
        self.set_seed(self.__seed)

    def set_seed(self, seed=None):

        if seed is not None:
            self.__seed = seed

        random.seed(self.__seed)
        np.random.seed(self.__seed)
        torch.manual_seed(self.__seed)

    def is_directed(self):
        '''
        Returns if the graph is directed or not

        :return
        '''

        return self.__directed

    def get_x0_s(self, standardize=True):
        '''
        :param standardize: If True, the initial positions are standardized
        :return: The initial positions of the sender nodes
        '''
        if standardize:
            return utils.standardize(self.__x0_s)
        else:
            return self.__x0_s

    def get_x0_r(self, standardize=True):
        '''
        :param standardize: If True, the initial positions nodes are standardized
        :return: The initial positions of the receiver nodes
        '''
        if standardize:
            return utils.standardize(self.__x0_r)
        else:
            return self.__x0_r

    def get_v_s(self, standardize=True):

        if standardize:
            return utils.standardize(self.__v_s)
        else:
            return self.__v_s

    def get_v_r(self, standardize=True):

        if standardize:
            return utils.standardize(self.__v_r)
        else:
            return self.__v_r

    def get_beta_s(self):

        return self.__beta_s

    def get_beta_r(self):

        return self.__beta_r

    def get_dim(self):

        return self.__dim

    def get_nodes_num(self):

        return self.__nodes_num

    def get_pairs_num(self):

        return (self.__nodes_num-  1) * self.__nodes_num // 2

    def get_bins_num(self):

        return self.__bins_num

    def get_bin_width(self):

        return self.__bin_width

    def get_seed(self):

        return self.__seed

    def get_verbose(self):

        return self.__verbose

    def get_device(self):

        return self.__device

    def get_init_time(self):

        return self.__init_time

    def get_last_time(self):

        return self.__last_time

    def get_init_states(self, pair_idx = None):

        if pair_idx is None:
            return self.__init_states
        else:
            return self.__init_states[pair_idx]

    def get_prior_lambda(self):
        '''
        Returns the scaling factor of covariance of the prior distribution
        '''
        return self.__prior_lambda

    def get_prior_sigma_s(self):
        '''
        Returns the noise parameter of the sender nodes for the covariance of the prior distribution
        '''
        return self.__prior_sigma_s

    def get_prior_sigma_r(self):
        '''
        Returns the noise parameter of the receiver nodes for the covariance of the prior distribution
        '''
        return self.__prior_sigma_r

    def get_prior_B_x0_c_s(self):
        '''
        Returns the parameter corresponding to the initial positions of sender nodes, 
        used in the construction of the node matrix, C 
        '''
        return self.__prior_B_x0_c_s

    def get_prior_B_x0_c_r(self):
        '''
        Returns the parameter corresponding to the initial positions of receiver nodes, 
        used in the construction of the node matrix, C 
        '''
        return self.__prior_B_x0_c_r

    def get_prior_B_ls_s(self):
        '''
        Returns the lengthscale parameter of sender nodes, used in the construction of bin matrix, B
        '''
        return self.__prior_B_ls_s

    def get_prior_B_ls_r(self):
        '''
        Returns the lengthscale parameter of receiver nodes, used in the construction of bin matrix, B
        '''
        return self.__prior_B_ls_r

    def get_prior_C_Q_s(self):
        '''
        Returns the factors of the sender node matrix, C (without softmax)
        '''
        return self.__prior_C_Q_s

    def get_prior_C_Q_r(self):
        '''
        Returns the factors of the receiver node matrix, C (without softmax)
        '''
        return self.__prior_C_Q_r

    def get_prior_k(self):
        '''
        Returns the latent dimension of the node matrix, C
        '''

        return self.__prior_C_Q_s.shape[1]

    def get_bin_bounds(self):
        '''
        Computes the bin bounds of the model
        :return: a vector of shape B+1
        '''
        bounds = self.__init_time + torch.cat((
            torch.as_tensor([self.get_init_time()], device=self.get_device()),
            torch.arange(start=1, end=self.get_bins_num()+1, device=self.get_device()) * self.get_bin_width()
        ))

        return bounds

    def get_bin_index(self, time_list: torch.Tensor):
        '''
        Computes the bin indices of given times

        :param time_list: a vector of shape L
        :return: an index and residual vectors of of shapes L
        '''

         # Compute the bin indices of the given time points
        bin_indices = utils.div(time_list, self.get_bin_width()).type(torch.long)
         # If there is a time equal to the last time, set its bin index to the last bin
        bin_indices[bin_indices == self.get_bins_num()] = self.get_bins_num() - 1

        return bin_indices

    def get_residual(self, time_list: torch.Tensor, bin_indices: torch.Tensor):

        '''
        Computes the residuals of given times

        :param time_list: a vector of shape L
        :return: an index and residual vectors of of shapes L
        '''

        # Compute the residual times
        residual_time = utils.remainder(time_list, self.get_bin_width())
        # If there is a time equal to the last time, set its resiudal time to bin width
        residual_time[bin_indices == self.get_bins_num()] = self.get_bin_width()

        return residual_time

    def get_rt_s(self, time_list: torch.Tensor, nodes: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        '''
        Computes the locations at given times based on the initial position and velocity tensors.
        
        :param time_list: a vector of shape L
        :param nodes: a vector of shape L
        :return: A matrix of shape L X D
        '''

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)
        residual_time = self.get_residual(time_list=time_list, bin_indices=bin_indices)

        # Get the initial position and velocity vectors
        x0 = self.get_x0_s(standardize=standardize)
        v = self.get_v_s(standardize=standardize)

        # Compute the displacement until the initial time of the corresponding bins
        cum_displacement = torch.cumsum(torch.cat((x0.unsqueeze(0),self.get_bin_width() * v)), dim=0)
        rt = cum_displacement[bin_indices, nodes, :]
        # Finally, add the displacement on the bin that time points lay on
        rt = rt + residual_time.view(-1, 1)*self.get_v_s(standardize=standardize)[bin_indices, nodes, :]
        return rt

    def get_rt_r(self, time_list: torch.Tensor, nodes: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        '''
        Computes the locations at given times based on the initial position and velocity tensors.
        
        :param time_list: a vector of shape L
        :param nodes: a vector of shape L
        :return: A matrix of shape L X D
        '''

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)
        residual_time = self.get_residual(time_list=time_list, bin_indices=bin_indices)

        # Get the initial position and velocity vectors
        x0 = self.get_x0_r(standardize=standardize)
        v = self.get_v_r(standardize=standardize)

        # Compute the displacement until the initial time of the corresponding bins
        cum_displacement = torch.cumsum(torch.cat((x0.unsqueeze(0),self.get_bin_width() * v)), dim=0)
        rt = cum_displacement[bin_indices, nodes, :]
        # Finally, add the displacement on the bin that time points lay on
        rt = rt + residual_time.view(-1, 1)*self.get_v_r(standardize=standardize)[bin_indices, nodes, :]
        return rt

    def get_beta_ij(self, pairs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        '''
        Computes the sum of the beta elements for given pair and states

        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param states: a vector of shape L
        :return: A vector of shape L
        '''

        beta_s = self.get_beta_s()
        beta_r = self.get_beta_r() if self.is_directed() else beta_s

        return beta_s[pairs[0], states] + beta_r[pairs[1], states]

    def get_delta_v(self, bin_indices: torch.Tensor, pairs: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        '''
        Computes the velocity diffrences for given bin indices and pairs.
        
        :param bin_indices: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param standardize: a boolean parameter to set the standardization of velocity vectors
        :return: A matrix of shape L X D
        '''

        v_s = self.get_v_s(standardize=standardize)
        v_r = self.get_v_r(standardize=standardize) if self.is_directed() else v_s

        return v_s[bin_indices, pairs[0], :] - v_r[bin_indices, pairs[1], :]

    def get_delta_rt(self, time_list: torch.Tensor, pairs: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        '''
        Computes the locations at given times based on the initial position and velocity tensors.
        
        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param standardize: a boolean parameter to set the standardization of the initial position and velocity vectors
        :return: A matrix of shape L X D
        '''

        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)
        residual_time = self.get_residual(time_list=time_list, bin_indices=bin_indices)
        
        # Get the initial position and velocity vectors
        x0_s = self.get_x0_s(standardize=standardize)
        v_s = self.get_v_s(standardize=standardize)

        if self.__directed:
            x0_r, v_r = self.get_x0_r(standardize=standardize), self.get_v_r(standardize=standardize)
        else:
            x0_r, v_r = x0_s, v_s

        # Compute the displacements
        cum_displacement_s = torch.cumsum(torch.cat((x0_s.unsqueeze(0), self.get_bin_width() * v_s)),  dim=0)
        cum_displacement_r = torch.cumsum(torch.cat((x0_r.unsqueeze(0), self.get_bin_width() * v_r)),  dim=0)

        # Select the bin indices and nodes
        delta_rt = cum_displacement_s[bin_indices, pairs[0], :] - cum_displacement_r[bin_indices, pairs[1], :]

        # Finally, add the displacement on the bin that time points lay on
        delta_rt += residual_time.view(-1, 1)*(v_s[bin_indices, pairs[0], :]-v_r[bin_indices, pairs[1], :])

        return delta_rt

    def get_log_intensity_at(self, time_list: torch.Tensor, pairs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        '''
        Computes the log of the intenstiy function for given times and pairs

        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :param states: a vector of shape L
        :return: A vector of shape L
        '''
        beta_ij = self.get_beta_ij(pairs=pairs, states=states)
        intensity = beta_ij + (2*states-1)*torch.norm(self.get_delta_rt(time_list=time_list, pairs=pairs), p=2, dim=1, keepdim=False)**2 

        return intensity

    def get_intensity_at(self, time_list: torch.Tensor, pairs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        '''
        Computes the intenstiy function for given times and pairs

        :param time_list: a vector of shape L
        :param pairs: a vector of shape 2 x L
        :return: A vector of shape L
        '''
        return torch.exp(self.get_log_intensity_at(time_list=time_list, pairs=pairs, states=states))

    def get_intensity_integral(self, time_list: torch.Tensor, pairs: torch.Tensor, delta_t: torch.Tensor, states: torch.Tensor, standardize: bool = True) -> torch.Tensor:
        '''
        Computes the negative log-likelihood function of the model

        :param time_list: a vector of shape L
        :param pairs: a matrix of shape 2 x L
        :param delta_t: a vector of shape L
        :param states: a vector of shape L
        :return:
        '''
        # Compute the bin indices and residul times of the given time points
        bin_indices = self.get_bin_index(time_list=time_list)

        # Get the position and velocity differences
        delta_r = self.get_delta_rt(time_list=time_list, pairs=pairs, standardize=standardize)
        delta_v = self.get_delta_v(bin_indices=bin_indices, pairs=pairs, standardize=standardize)

        # Compute the beta sums
        beta_ij = self.get_beta_ij(pairs=pairs, states=states)

        norm_delta_r = torch.norm(delta_r, p=2, dim=1, keepdim=False)
        norm_delta_v = torch.norm(delta_v, p=2, dim=1, keepdim=False) + utils.EPS

        inv_norm_delta_v = 1.0 / norm_delta_v
        delta_r_v = (delta_r * delta_v).sum(dim=1, keepdim=False)
        r = delta_r_v * inv_norm_delta_v

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self.__device)) * inv_norm_delta_v
        term1 = torch.exp(beta_ij - (2*states-1)*(r**2) + (2*states-1)*(norm_delta_r ** 2))

        term2_u = (1-states)*torch.erf(delta_t * norm_delta_v + r) + states*utils.erfi_approx(delta_t * norm_delta_v + r)
        term2_l = (1-states)*torch.erf(r) + states*utils.erfi_approx(r)

        return (term0 * term1 * (term2_u - term2_l))

    def get_nll(self, pairs: torch.Tensor, states:torch.Tensor, borders:torch.Tensor, is_event:torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        '''
        Computes the negative log-likelihood function of the model

        :param pairs: a matrix of shape 2 x L
        :param states: a vector of shape L
        :param borders: a vector of shape L
        :param is_event: a vector of shape L
        :param delta_t: a vector of shape L
        :return:
        '''
        
        non_integral_term = self.get_log_intensity_at(
            time_list=borders[is_event==True], pairs=pairs[:, is_event==True], states=states[is_event==True]
        ).sum()

        integral_term = -self.get_intensity_integral(
            time_list=borders, pairs=pairs.type(torch.long), delta_t=delta_t, states=states.type(torch.long)
        ).sum()

        return -(non_integral_term + integral_term)

    def get_neg_log_prior(self, nodes: torch.Tensor, compute_R_factor_inv: bool = True) -> torch.float:

        # Define the final dimension size
        final_dim = self.get_nodes_num() * (self.get_bins_num()+1) * self.get_dim()

        # Covariance scaling coefficient
        lambda_sq = self.get_prior_lambda()**2

        # Noise for the Gaussian process
        sigma_sq_inv_s = 1.0 / (self.get_prior_sigma_s()**2)
        if self.is_directed():
            sigma_sq_inv_r = 1.0 / (self.get_prior_sigma_r()**2)

        # Get the bin bounds
        bounds = self.get_bin_bounds()

        # Get the middle time points of the bins
        middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

        # B x B matrix
        B_factor_s = prior.get_B_factor(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, prior_B_x0_c=self.get_prior_B_x0_c_s(), prior_B_ls=self.get_prior_B_ls_s()
        )
        # N x K matrix where K is the community size
        C_factor_s = prior.get_C_factor(prior_C_Q=self.get_prior_C_Q_s())
        # D x D matrix
        D_factor_s = prior.get_D_factor(dim=self.get_dim())

        if self.is_directed():
            B_factor_r = prior.get_B_factor(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds, prior_B_x0_c=self.get_prior_B_x0_c_r(), prior_B_ls=self.get_prior_B_ls_r()
            )
            C_factor_r = prior.get_C_factor(prior_C_Q=self.get_prior_C_Q_r())
            D_factor_r = prior.get_D_factor(dim=self.get_dim())

        # Compute the Kf factor
        Kf_s = torch.kron(B_factor_s.contiguous(), torch.kron(torch.index_select(C_factor_s, dim=0, index=nodes), D_factor_s))
        if self.is_directed():
            Kf_r = torch.kron(B_factor_r.contiguous(), torch.kron(torch.index_select(C_factor_r, dim=0, index=nodes), D_factor_r))

        # Compute the R factor if not provided
        if compute_R_factor_inv is True:

            # R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
            R_factor_s = prior.get_R_factor(
                dim=self.get_prior_k() * (self.get_bins_num()+1) * self.get_dim(), sigma=self.get_prior_sigma_s(), 
                B_factor=B_factor_s, C_factor=C_factor_s, D_factor=D_factor_s
            )
            R_factor_inv_s = torch.inverse(R_factor_s)
            self.__prior_R_factor_inv_s = R_factor_inv_s

            if self.is_directed():
                R_factor_r = prior.get_R_factor(
                    dim=self.get_prior_k() * (self.get_bins_num()+1) * self.get_dim(), sigma=self.get_prior_sigma_r(), 
                    B_factor=B_factor_r, C_factor=C_factor_r, D_factor=D_factor_r
                )
                R_factor_inv_r = torch.inverse(R_factor_r)
                self.__prior_R_factor_inv_r = R_factor_inv_r

        else:
            R_factor_inv_s = self.__prior_R_factor_inv_s

            if self.is_directed():
                R_factor_inv_r = self.__prior_R_factor_inv_r

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
        mahalanobis_term1_s = sigma_sq_inv_s * x0v_s.pow(2).sum(-1)
        mahalanobis_term2_s = (sigma_sq_inv_s * x0v_s @ Kf_s @ R_factor_inv_s.T).pow(2).sum(-1)
        m_s = (1.0 / lambda_sq) * ( mahalanobis_term1_s - mahalanobis_term2_s )
        if self.is_directed():
            mahalanobis_term1_r = sigma_sq_inv_r * x0v_r.pow(2).sum(-1)
            mahalanobis_term2_r = (sigma_sq_inv_r * x0v_r @ Kf_r @ R_factor_inv_r.T).pow(2).sum(-1)
            m_r = (1.0 / lambda_sq) * ( mahalanobis_term1_r - mahalanobis_term2_r )

        # Computation of the log determinant
        # It uses Matrix Determinant Lemma: log|D + Kf @ Kf.T| = log|R| + log|D|,
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        log_det_s = -2 * R_factor_inv_s.diag().log().sum() + final_dim*(lambda_sq.log() - sigma_sq_inv_s.log())
        if self.is_directed():
            log_det_r = -2 * R_factor_inv_r.diag().log().sum() + final_dim*(lambda_sq.log() - sigma_sq_inv_r.log())

        # Compute the negative log-likelihood
        log_prior_s = -0.5 * (final_dim * utils.LOG2PI + log_det_s + m_s)
        if self.is_directed():
            log_prior_r = -0.5 * (final_dim * utils.LOG2PI + log_det_r + m_r)

        neg_log_prior = -log_prior_s
        if self.is_directed():
            neg_log_prior += -log_prior_r

        return -neg_log_prior.squeeze()
