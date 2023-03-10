import torch
from utils.common import EPS


def rbf(t1: torch.Tensor, t2: torch.Tensor, ls: torch.Tensor):
    '''
    Compute the RBF kernel

    param: t1: input values
    param: t3: input values
    param ls: length scale parameter
    '''
    time_mat = t1 - t2.T
    return torch.exp(-(time_mat**2 / (2.0*ls**2)))

def get_B(bin_centers1: torch.Tensor, bin_centers2: torch.Tensor, prior_B_x0_c: torch.Tensor, prior_B_ls: torch.Tensor, noise: float = 10*EPS) -> torch.Tensor:
    '''
    Computes the bin matrix, B, of shape (B+1) x (B+1)

    :param bin_centers1: bin centers
    :param bin_centers2: bin centers
    :param prior_B_x0_c: parameter corresponding for the initial positions
    :param prior_B_ls: length-scale parameter of the RBF kernel used for the velocities
    :return:
    '''

    # Construct the parameter for the initial position
    prior_B_x0_c_sq = prior_B_x0_c**2

    # Construct the parameters for the velocities
    kernel = rbf(t1=bin_centers1, t2=bin_centers2, ls=prior_B_ls)

    # Combine the entry required for x0 with the velocity vectors covariance
    kernel = torch.block_diag(prior_B_x0_c_sq, kernel)

    # Add a constant term to get rid of computational problems and singularity
    kernel = kernel + noise*torch.eye(n=kernel.shape[0], m=kernel.shape[1])

    return kernel


def get_B_factor(bin_centers1: torch.Tensor, bin_centers2: torch.Tensor, prior_B_x0_c: torch.Tensor, prior_B_ls: torch.Tensor):
    '''
    Computes the bin matrix factor, B, of shape (B+1) x (B+1)

    :param bin_centers1: bin centers
    :param bin_centers2: bin centers
    :param prior_B_x0_c: parameter corresponding for the initial positions
    :param prior_B_ls: length-scale parameter of the RBF kernel used for the velocities
    :return:
    '''
    # B x B lower triangular matrix
    return torch.linalg.cholesky(get_B(bin_centers1, bin_centers2, prior_B_x0_c, prior_B_ls))


def get_C_factor(prior_C_Q):
    '''
    Computes the factor of the node matrix of shape N x K
    
    :param prior_C_Q: hyper-parameter to construct factor matrix
    :return:
    '''

    # N x K matrix
    return torch.softmax(prior_C_Q, dim=1)


def get_D_factor(dim):
    '''
    Computes the factor of the dimension matrix of shape D x D

    :param dim: dimension
    :return:
    '''
    # D x D matrix
    return torch.eye(dim)

def get_R(dim, sigma, B_factor, C_factor, D_factor):
    '''
    Computes the capacitance matrix, R
    :param dim: dimension
    :param sigma: noise parameter
    :param B_factor: the factor of the bin matrix of shape (B+1)x(B+1)
    :param C_factor: the factor of the node matrix of shape N x K
    :param D_factor: the dimension matrix of shape DxD
    :return:
    '''

    R = torch.eye(dim) + (1./sigma**2)*torch.kron(B_factor.T @ B_factor, torch.kron(C_factor.T @ C_factor, D_factor.T @ D_factor))

    return R

def get_R_factor(dim, sigma, B_factor, C_factor, D_factor):
    '''
    Computes the factor of the capacitance matrix R which is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
    :param dim: dimension
    :param sigma: noise parameter
    :param B_factor: the factor of the bin matrix of shape (B+1)x(B+1)
    :param C_factor: the factor of the node matrix of shape N x K
    :param D_factor: the dimension matrix of shape DxD
    :return:
    '''

    R_factor = torch.linalg.cholesky(get_R(dim, sigma, B_factor, C_factor, D_factor))

    return R_factor