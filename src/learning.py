import sys
import math
import torch
import time
import utils
from src.base import BaseModel
from src.sampler import BatchSampler


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, nodes_num: int, directed: bool, signed: bool, bins_num: int, dim: int,
                 prior_lambda: float = 1e5, k=10,
                 device: torch.device = 'cpu', verbose: bool = False, seed: int = 19):

        utils.set_seed(seed)

        super(LearningModel, self).__init__(
            x0_s=torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False),
            x0_r=torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False) if directed else None,
            v_s=torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
            v_r=torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False) if directed else None,
            beta_s=torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, 2), device=device), requires_grad=False),
            beta_r=torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, 2), device=device), requires_grad=False) if directed else None,
            prior_lambda=torch.as_tensor(prior_lambda, dtype=torch.float, device=device),
            prior_sigma_s=torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False),
            prior_sigma_r=torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False) if directed else None,
            prior_B_x0_logit_c_s=torch.nn.Parameter(torch.special.logit(torch.rand(size=(1, ), device=device)), requires_grad=False),
            prior_B_x0_logit_c_r=torch.nn.Parameter(torch.special.logit(torch.rand(size=(1, ), device=device)), requires_grad=False) if directed else None,
            prior_B_ls_s=torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False),
            prior_B_ls_r=torch.nn.Parameter(2. * torch.rand(size=(1, ), device=device) - 1, requires_grad=False) if directed else None,
            prior_C_Q_s=torch.nn.Parameter(2. * torch.rand(size=(nodes_num, k), device=device) - 1,  requires_grad=False),
            prior_C_Q_r=torch.nn.Parameter(2. * torch.rand(size=(nodes_num, k), device=device) - 1,  requires_grad=False)  if directed else None,
            prior_R_factor_inv_s = None, prior_R_factor_inv_r = None,
            directed=directed,
            signed=signed,
            device=device,
            verbose=verbose,
            seed=seed
        )

        # Learning parameters
        self.__lp = "sequential"  # Learning procedure
        self.__optimizer = torch.optim.Adam  # Optimizer

        # - The following parameters are set in the learn() method
        self.__masked_pair_indices = None
        self.__min_time = None
        self.__max_time = None
        self.__lr = None  # Learning rate
        self.__batch_size = None   # Batch size
        self.__epoch_num = None  # Number of epochs
        self.__steps_per_epoch = None  # Number of steps per epoch

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, prior_grad=None):
        '''
        Set the gradient status of the model parameters
        :param beta_grad: The gradient for the bias terms
        :param x0_grad: The gradient for the initial positions
        :param v_grad: The gradient for the velocities
        '''

        # Set the gradient of the bias terms
        if beta_grad is not None:
            self.get_beta_s().requires_grad = beta_grad
            if self.is_directed():
                self.get_beta_r().requires_grad = beta_grad

        # Set the gradient of the initial positions
        if x0_grad is not None:
            self.get_x0_s(standardize=False).requires_grad = x0_grad
            if self.is_directed():
                self.get_x0_r(standardize=False).requires_grad = x0_grad

        # Set the gradient of the velocities
        if v_grad is not None:
            self.get_v_s(standardize=False).requires_grad = v_grad
            if self.is_directed():
                self.get_v_r(standardize=False).requires_grad = v_grad
                
        # Set the gradient of the all prior parameters
        if prior_grad is not None:
            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = prior_grad

    def learn(self, dataset, lr: float = 0.1, batch_size: int = 0, epoch_num: int = 100, steps_per_epoch=1,
              masked_dataset=None, log_file_path=None):

        # Set the learning parameters
        self.__lr = lr
        self.__batch_size = self.get_nodes_num() if batch_size == 0 else batch_size
        self.__epoch_num = epoch_num
        self.__steps_per_epoch = steps_per_epoch

        # Select the masked pairs
        self.__masked_pair_indices = utils.matIdx2flatIdx(
            i=masked_dataset.get_edges(0), j=masked_dataset.get_edges(0), n=self.get_nodes_num(),
            is_directed=self.is_directed(), dtype=torch.long
        ).unique().to(self.get_device()) if masked_dataset is not None else None

        # Scale the edge times to [0, 1]
        edge_times = dataset.get_times()
        self.__min_time = edge_times.min()
        self.__max_time = edge_times.max()
        edge_times = (edge_times - self.__min_time) / (self.__max_time - self.__min_time)

        # Define the batch sampler
        bs = BatchSampler(
            edges=dataset.get_edges().to(self.get_device()), edge_times=edge_times.to(self.get_device()),
            edge_states=dataset.get_states().to(self.get_device()),
            bin_bounds=self.get_bin_bounds(), nodes_num=self.get_nodes_num(), batch_size=self.__batch_size, 
            directed=self.is_directed(), device=self.get_device(), seed=self.get_seed()
        )

        # Check the learning procedure
        if self.__lp == "sequential":

            # Run sequential learning
            loss, nll = self.__sequential_learning(bs=bs)

        else:

            raise Exception(f"Unknown learning procedure: {self.__lp}")

        # Save the loss if the loss file path was given
        if log_file_path is not None:
            
            with open(log_file_path, 'w') as f:
                for batch_losses, nll_losses in zip(loss, nll):
                    f.write(f"Loss: {' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")
                    f.write(f"Nll: {' '.join('{:.3f}'.format(loss) for loss in nll_losses)}\n")

        return loss, nll

    def __sequential_learning(self, bs: BatchSampler):

        if self.get_verbose():
            print("+ Training started (Procedure: Sequential Learning).")

        # Define the optimizers and parameter group names
        self.group_optimizers = []
        self.param_groups = [["v"], ["x0", "v"], ["beta", "x0", "v", "prior"] ]
        self.group_epoch_weights = [1.0, 1.0, 1.0]

        # For each group of parameters, add a new optimizer
        for current_group in self.param_groups:
            
            # Set the gradients to True
            self.__set_gradients(**{f"{param_name}_grad": True for param_name in current_group})
        
            # Add a new optimizer
            self.group_optimizers.append(self.__optimizer(self.parameters(), lr=self.__lr))
            
            # Set the gradients to False
            self.__set_gradients(**{f"{param_name}_grad": False for param_name in current_group})

        # Determine the number of epochs for each parameter group
        group_epoch_counts = (self.__epoch_num * torch.cumsum(
            torch.as_tensor([0.] + self.group_epoch_weights, device=self.get_device(), dtype=torch.float), dim=0
        ) / sum(self.group_epoch_weights)).type(torch.int)
        group_epoch_counts = group_epoch_counts[1:] - group_epoch_counts[:-1]

        # Run the epochs
        loss, nll = [], []
        epoch_num = 0
        for current_epoch_count, optimizer, current_group in zip(group_epoch_counts, self.group_optimizers, self.param_groups):

            # Set the gradients to True
            self.__set_gradients(**{f"{param_name}_grad": True for param_name in current_group})

            # Run the epochs
            for _ in range(current_epoch_count):
                # Run one epoch
                epoch_loss, epoch_nll = self.__train_one_epoch(bs=bs, epoch_num=epoch_num, optimizer=optimizer)
                loss.append(epoch_loss)
                nll.append(epoch_nll)

                # Increase the epoch number by one
                epoch_num += 1

            # Set the gradients to False
            self.__set_gradients(**{f"{param_name}_grad": False for param_name in current_group})

        return loss, nll

    def __train_one_epoch(self, bs: BatchSampler, epoch_num: int, optimizer: torch.optim.Optimizer):

        init_time = time.time()

        total_batch_loss = 0.
        epoch_loss, epoch_nll = [], []
        for batch_num in range(self.__steps_per_epoch):

            batch_nll, batch_nlp = self.__train_one_batch(bs=bs, batch_num=batch_num)
            batch_loss = batch_nll + batch_nlp

            epoch_loss.append(batch_loss)
            epoch_nll.append(batch_nll)

            total_batch_loss += batch_loss

        # Set the gradients to 0
        optimizer.zero_grad()

        # Backward pass
        total_batch_loss.backward()

        # Perform a step
        optimizer.step()

        # Get the average epoch loss
        avg_loss = total_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(avg_loss):
            print(f"\t- Epoch loss is {avg_loss}, stopping training")
            sys.exit(1)

        if self.get_verbose() and (epoch_num % 10 == 0 or epoch_num == self.__epoch_num - 1):
            time_diff = time.time() - init_time
            print("\t- Epoch = {} | Avg. Loss/train: {} | Elapsed time: {:.2f}".format(epoch_num, avg_loss, time_diff))

        return epoch_loss, epoch_nll

    def __train_one_batch(self, bs: BatchSampler, batch_num: int) -> tuple[torch.Tensor, torch.Tensor]:

        self.train()

        # Sample a batch
        batch_nodes, expanded_pairs, expanded_times, expanded_states, is_edge, delta_t = bs.sample()

        if self.__masked_pair_indices is not None:

            selected_indices = (utils.matIdx2flatIdx(
                i=expanded_pairs[0], j=expanded_pairs[1], n=self.get_nodes_num(), is_directed=self.__batch_size
            ).unsqueeze(1) == self.__masked_pair_indices).any(1) == False

            expanded_pairs = expanded_pairs[:, selected_indices]
            expanded_times = expanded_times[selected_indices]
            expanded_states = expanded_states[selected_indices]
            is_edge = is_edge[selected_indices]
            delta_t = delta_t[selected_indices]

        # Compute the R factor inverse if the batch number is 0
        compute_R_factor_inv = True if batch_num == 0 else False

        # Finally, compute the negative log-likelihood and the negative log-prior for the batch
        batch_nll, batch_nlp = self.forward(
            nodes=batch_nodes, pairs=expanded_pairs, times=expanded_times, states=expanded_states,
            is_edge=is_edge, delta_t=delta_t, compute_R_factor_inv=compute_R_factor_inv
        )

        # Divide the batch loss by the number of all possible pairs
        average_batch_nll = batch_nll / float(self.__batch_size * (self.__batch_size - 1))
        average_batch_nlp = batch_nlp / float(self.__batch_size * (self.__batch_size - 1))
        if not self.is_directed():
            average_batch_nll /= 2.
            average_batch_nlp /= 2.

        return average_batch_nll, average_batch_nlp

    def forward(self, nodes: torch.Tensor, pairs: torch.LongTensor, times: torch.FloatTensor,
                states: torch.LongTensor, is_edge: torch.BoolTensor, delta_t: torch.FloatTensor,
                compute_R_factor_inv=True) -> tuple[torch.Tensor, torch.Tensor]:

        # Get the negative log-likelihood
        nll = self.get_nll(pairs=pairs, times=times, states=states, is_edge=is_edge, delta_t=delta_t)

        # Get the negative log-prior and the R-factor inverse
        nlp = self.get_neg_log_prior(nodes=nodes, compute_R_factor_inv=compute_R_factor_inv)

        return nll, nlp

    def save(self, path):

        if self.get_verbose():
            print(f"+ Model file is saving.")
            print(f"\t- Target path: {path}")

        kwargs = {
            'nodes_num': self.get_nodes_num(),
            'directed': self.is_directed(),
            'signed': self.is_signed(),
            'bins_num': self.get_bins_num(),
            'dim': self.get_dim(),
            'prior_lambda': self.get_prior_lambda(), 
            'k': self.get_prior_k(),
            'verbose': self.get_verbose(), 
            'seed': self.get_seed()
        }
        
        torch.save([kwargs, self.state_dict()], path)

        if self.get_verbose():
            print(f"\t- Completed.")

