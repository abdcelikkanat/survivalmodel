import sys
import math
import torch
import time
import utils
from src.base import BaseModel
from src.sampler import BatchSampler


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, nodes_num: int, directed: bool, bins_num: int, dim: int,
                 prior_lambda: float = 1e5, k=10, lr: float = 0.1, batch_size: int = None, epoch_num: int = 100, steps_per_epoch=1, 
                 device: torch.device = 'cpu', verbose: bool = False, seed: int = 19):

        utils.set_seed(seed)

        super(LearningModel, self).__init__(
            x0_s = torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False), 
            x0_r = torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False) if directed else None,
            v_s = torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
            v_r = torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False) if directed else None,
            beta_s = torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, 2), device=device), requires_grad=False),
            beta_r = torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, 2), device=device), requires_grad=False) if directed else None,
            prior_lambda = torch.as_tensor(prior_lambda, dtype=torch.float, device=device),
            prior_sigma_s = torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False), 
            prior_sigma_r = torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False) if directed else None,
            prior_B_x0_logit_c_s = torch.nn.Parameter(torch.special.logit(torch.rand(size=(1, ), device=device)), requires_grad=False), 
            prior_B_x0_logit_c_r = torch.nn.Parameter(torch.special.logit(torch.rand(size=(1, ), device=device)), requires_grad=False) if directed else None,
            prior_B_ls_s = torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False), 
            prior_B_ls_r = torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False) if directed else None,
            prior_C_Q_s = torch.nn.Parameter(2. * torch.rand(size=(nodes_num, k), device=device) - 1,  requires_grad=False), 
            prior_C_Q_r = torch.nn.Parameter(2. * torch.rand(size=(nodes_num, k), device=device) - 1,  requires_grad=False)  if directed else None,
            prior_R_factor_inv_s = None, prior_R_factor_inv_r = None,
            directed=directed,
            device=device,
            verbose=verbose,
            seed=seed
        )

        # Optimization parameters
        self.__lp = "sequential" # Learning procedure
        self.__lr = lr # Learning rate
        self.__batch_size = nodes_num if batch_size is None else batch_size # Batch size
        self.__epoch_num = epoch_num # Number of epochs
        self.__steps_per_epoch = steps_per_epoch # Number of steps per epoch
        self.__optimizer = torch.optim.Adam # Optimizer

        self.__min_time = None
        self.__max_time = None

    def __set_gradients(self, param_name, value):
        '''
        Set the gradient status of the model parameters
        :param param_name: Parameter name
        :param value: The value of the gradient status
        '''

        if param_name == 'beta':

            self.get_beta_s().requires_grad = value
            if self.is_directed():
                self.get_beta_r().requires_grad = value

        elif param_name == 'x0':

            self.get_x0_s(standardize=False).requires_grad = value
            if self.is_directed():
                self.get_x0_r().requires_grad = value

        elif param_name[0] == 'v':

            bin_idx = int(param_name[1])

            self.get_v_s(standardize=False).requires_grad = value
            if self.is_directed():
                self.get_v_r().requires_grad = value

        elif param_name == 'prior':

            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = value

        else:

            raise Exception(f"Unknown parameter: {param_name}")

    def learn(self, dataset, loss_file_path=None):

        # Scale the edge times to [0, 1]
        edge_times = dataset.get_times()
        self.__min_time = edge_times.min()
        self.__max_time = edge_times.max()
        edge_times = (edge_times - self.__min_time) / (self.__max_time - self.__min_time)

        # Define the batch sampler
        bs = BatchSampler(
            edges=dataset.get_edges(), edge_times=edge_times, edge_states=dataset.get_states(),
            nodes_num=self.get_nodes_num(), batch_size=self.__batch_size,
            directed=self.is_directed(), device=self.get_device(), seed=self.get_seed()
        )

        # Check the learning procedure
        if self.__lp == "sequential":

            # Run sequential learning
            loss, nll = self.__sequential_learning(bs=bs)

        else:

            raise Exception(f"Unknown learning procedure: {self.__lp}")

        # Save the loss if the loss file path was given
        if loss_file_path is not None:
            
            with open(loss_file_path, 'w') as f:
                for batch_losses, nll_losses in zip(loss, nll):
                    f.write(f"Loss: {' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")
                    f.write(f"Nll: {' '.join('{:.3f}'.format(loss) for loss in nll_losses)}\n")

        return loss, nll

    def __sequential_learning(self, bs: BatchSampler):

        if self.get_verbose():
            print("- Training started (Procedure: Sequential Learning).")

        # Define the parameter order for the optimization procedure
        # [x0], [x0, v1, v2], [x0, v1, v2, v3], ...
        param_groups = [["x0", "v1"]] + [[f"v{b}"] for b in range(2, self.get_bins_num()+1)]
        # Define the parameter weights for the optimization procedure
        param_weights = torch.arange(1, self.get_bins_num()+1).to(torch.float)  #[1]*self.get_bins_num()

        # Compute the epoch counts for each parameter group
        # group_epoch_counts = torch.arange(
        #     0, self.__epoch_num, step=int(self.__epoch_num // sum(param_weights)),
        #     dtype=torch.int, device=self.get_device()
        # ).diff(append=torch.tensor([self.__epoch_num], dtype=torch.int, device=self.get_device()))
        group_epoch_counts = (self.__epoch_num * param_weights / param_weights.sum()).to(torch.int)
        print("----", group_epoch_counts)

        # Define the optimizers and parameter group names
        group_optimizers = []
        # Since we set the gradient of tensors to False in defining them, we need to set them to True one by one
        for current_group in param_groups:
            # Set the gradients to True
            for param_name in current_group:
                self.__set_gradients(param_name, True)
            # Dedicate a distinct optimizer to the current parameter group
            group_optimizers.append(self.__optimizer(self.parameters(), lr=self.__lr))
        # Set the gradients to False again
        for current_group in param_groups:
            for param_name in current_group:
                self.__set_gradients(param_name, False)

        # Run the epochs
        loss, nll = [], []
        epoch_num = 0
        for current_epoch_count, optimizer, current_group in zip(group_epoch_counts, group_optimizers, param_groups):
            # Set the gradients to True again here
            for param_name in current_group:
                self.__set_gradients(param_name, True)

            # if v is in the current group, then set the max edge time
            max_edge_time = None
            for param in current_group:
                if param[0] == 'v':
                    max_edge_time = self.get_bin_width()*int(param[1:])
                    # print("group:", current_group, f"Max edge time: {max_edge_time}")
                    # print(f"Max edge time: {max_edge_time}", "epoch:", epoch_num, "current_epoch_count", current_epoch_count)

            # Run the epochs
            for _ in range(current_epoch_count):
                epoch_loss, epoch_nll = self.__train_one_epoch(
                    bs=bs, max_edge_time=max_edge_time, epoch_num=epoch_num, optimizer=optimizer
                )
                loss.append(epoch_loss)
                nll.append(epoch_nll)

                # Increase the epoch number by one
                epoch_num += 1

        return loss, nll

    def __train_one_epoch(self, bs: BatchSampler, max_edge_time: float, epoch_num: int, optimizer: torch.optim.Optimizer):

        init_time = time.time()

        total_batch_loss = 0
        epoch_loss, epoch_nll = [], []
        for batch_num in range(self.__steps_per_epoch):

            batch_nll, batch_nlp = self.__train_one_batch(bs=bs, max_edge_time=max_edge_time, batch_num=batch_num)
            batch_loss = batch_nll + batch_nlp

            epoch_loss.append(batch_loss)
            epoch_nll.append(batch_nll)

            total_batch_loss += batch_loss

        # Set the gradients to 0
        optimizer.zero_grad()

        # Backward pass
        total_batch_loss.backward()

        bin_idx = self.get_bin_index(torch.as_tensor([max_edge_time]))
        if max_edge_time == 1.:
            bin_idx = self.get_bins_num()
        self.get_v_s(standardize=False).grad[bin_idx:, :, :] = 0
        # self.get_v_s(standardize=False).grad[:bin_idx-1, :, :] = 0

        # Perform a step
        optimizer.step()

        # Get the average epoch loss
        avg_loss = total_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(avg_loss):
            print(f"- Epoch loss is {avg_loss}, stopping training")
            sys.exit(1)

        if self.get_verbose() and (epoch_num % 10 == 0 or epoch_num == self.__epoch_num - 1):
            time_diff = time.time() - init_time
            print("\t+ Epoch = {} | Avg. Loss/train: {} | Elapsed time: {:.2f}".format(epoch_num, avg_loss, time_diff))

        return epoch_loss, epoch_nll

    def __train_one_batch(self, bs: BatchSampler, max_edge_time: float, batch_num: int):

        self.train()

        # Sample a batch
        batch_nodes, batch_pairs, batch_edges, batch_edge_times, batch_edge_states = bs.sample(max_edge_time)

        # Compute the R factor inverse if the batch number is 0
        compute_R_factor_inv = True if batch_num == 0 else False

        # Finally, compute the negative log-likelihood and the negative log-prior for the batch
        batch_nll, batch_nlp = self.forward(
            nodes=batch_nodes, pairs=batch_pairs, batch_edges=batch_edges, 
            batch_edge_times=batch_edge_times, batch_states=batch_edge_states, 
            max_edge_time=max_edge_time, compute_R_factor_inv=compute_R_factor_inv
        )

        # Divide the batch loss by the number of all possible pairs
        average_batch_nll = batch_nll / float(batch_pairs.shape[1])
        average_batch_nlp = batch_nlp / float(batch_pairs.shape[1])

        return average_batch_nll, average_batch_nlp

    def forward(self, nodes: torch.Tensor, pairs: torch.LongTensor, batch_edges:torch.LongTensor, 
                batch_edge_times: torch.FloatTensor, batch_states: torch.LongTensor,
                max_edge_time: float, compute_R_factor_inv=True):

        # Get the negative log-likelihood
        nll = self.get_nll(
            pairs=pairs, edges=batch_edges, edge_times=batch_edge_times,
            edge_states=batch_states, max_edge_time=max_edge_time
        )

        # Get the negative log-prior and the R-factor inverse
        nlp = 0 #self.get_neg_log_prior(nodes=nodes, compute_R_factor_inv=compute_R_factor_inv)

        return nll, nlp

    def save(self, path):

        if self.get_verbose():
            print(f"- Model file is saving.")
            print(f"\t+ Target path: {path}")

        kwargs = {
            'nodes_num': self.get_nodes_num(),
            'directed': self.is_directed(),
            'bins_num': self.get_bins_num(),
            'dim': self.get_dim(),
            'prior_lambda': self.get_prior_lambda(), 
            'k': self.get_prior_k(),
            'lr': self.__lr,
            'batch_size':  self.__batch_size,
            'epoch_num': self.__epoch_num,
            'steps_per_epoch': self.__steps_per_epoch,
            'verbose': self.get_verbose(), 
            'seed': self.get_seed()
        }

        # # Scale the parameters
        # self.state_dict()['_BaseModel__v_s'].mul_( 1. / (self.__max_time - self.__min_time) )
        # if self.is_directed():
        #     self.state_dict()['_BaseModel__v_r'].mul_( 1. / (self.__max_time - self.__min_time) )
        
        torch.save([kwargs, self.state_dict()], path)

        if self.get_verbose():
            print(f"\t+ Completed.")

