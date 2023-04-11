import sys
import math
import torch
from src.base import BaseModel
import time
import utils


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data: tuple, nodes_num: int, bins_num: int, dim: int, k=10, prior_lambda: float = 1e4, directed = False,
                 lr: float = 0.1, batch_size: int = None, epoch_num: int = 100, steps_per_epoch=10, 
                 device: torch.device = None, verbose: bool = False, seed: int = 19):

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
            init_states=torch.nn.Parameter(torch.zeros(size=((nodes_num-1)*nodes_num//2, ), device=device), requires_grad=False),
            directed=directed,
            device=device,
            verbose=verbose,
            seed=seed
        )

        # Optimization parameters
        self.__lp = "sequential" # Learning procedure
        self.__lr = lr # Learning rate
        self.__epoch_num = epoch_num # Number of epochs
        self.__batch_size = nodes_num if batch_size is None else batch_size # Batch size
        self.__steps_per_epoch = steps_per_epoch # Number of steps per epoch
        self.__optimizer = torch.optim.Adam # Optimizer

        # Define the model variables
        self.__sparse_border_mat = None
        self.__mat_pairs = None
        self.__border_times = None
        self.__is_event = None
        self.__states = None 
        self.__delta_t = None
        self.__max_row_len = None
        
        # Compute the model variables
        self.__sparse_border_mat, self.__mat_pairs, self.__border_pairs, self.__border_times, self.__is_event, self.__states, self.__delta_t, self.__max_row_len = self.__initialize(data=data)


    def __initialize(self, data):
        '''
        Compute the model variables

        :param data: A tuple of (pairs, times, states)
        '''

        pairs, times, states = data

        pairs = torch.as_tensor(pairs, dtype=torch.long, device=self.get_device()).T
        times = torch.as_tensor(times, dtype=torch.float, device=self.get_device())

        # Add the bin times for each pair and reconstruct the pairs and times tensors
        border_pairs = torch.hstack((
            torch.repeat_interleave(torch.triu_indices(
                self.get_nodes_num(), self.get_nodes_num(), offset=1, dtype=torch.long, device=self.get_device()
            ), repeats=self.get_bins_num(), dim=1),
            pairs
        ))

        border_times = torch.hstack((
            self.get_bin_bounds()[:-1].unsqueeze(0).expand(self.get_pairs_num(), self.get_bins_num()).flatten(),
            times
        ))

        # Construct a tensor indicating if the corresponding time is an event/link or not
        is_event = torch.hstack((
            torch.zeros(len(border_times)-len(times),  dtype=torch.bool, device=self.get_device()),
            torch.ones_like(times, dtype=torch.bool, device=self.get_device())
        ))

        # Sort all pairs, times and is_event vector
        sorted_indices = torch.argsort(border_times)
        border_pairs = border_pairs[:, sorted_indices]
        border_times = border_times[sorted_indices]
        is_event = is_event[sorted_indices]

        #### FIX - FIX - FIX ####
        mat_row_indices = utils.pairIdx2flatIdx(border_pairs[0], border_pairs[1], n=self.get_nodes_num()).type(torch.long)
        counts = [0]*self.get_pairs_num() #torch.zeros(self.number_of_pairs(), dtype=torch.long, device=self.get_device())
        mat_col_indices = []
        for r in mat_row_indices:
            mat_col_indices.append(counts[r])
            counts[r] += 1
        mat_col_indices = torch.as_tensor(mat_col_indices, dtype=torch.long, device=self.get_device())
        max_row_len = max(counts) + 1
        mat_pairs = torch.vstack((mat_row_indices, mat_col_indices))
        #### FIX - FIX - FIX ####

        # Find the corresponding border times of the chosen times
        sparse_border_mat = torch.sparse_coo_tensor(mat_pairs, values=border_times, size=(self.get_pairs_num(), max_row_len))

        states = torch.sparse.mm(
            torch.sparse_coo_tensor(mat_pairs, values=is_event.type(torch.int), size=(self.get_pairs_num(), max_row_len)), 
            torch.triu(torch.ones(size=(max_row_len, max_row_len), dtype=torch.int, device=self.get_device()), diagonal=0)
        )[mat_row_indices, mat_col_indices]
        states[states % 2 == 0] = 0
        states[states > 0] = 1

        delta_t = torch.sparse.mm(
            torch.sparse_coo_tensor(mat_pairs, values=border_times, size=(self.get_pairs_num(), max_row_len)), 
            -torch.diag(torch.ones(max_row_len, dtype=torch.float, device=self.get_device()), diagonal=0) + 
            torch.diag(torch.ones(max_row_len-1, dtype=torch.float, device=self.get_device()), diagonal=-1) 
        )[mat_row_indices, mat_col_indices]
        delta_t[delta_t < 0] = self.get_last_time() + delta_t[delta_t < 0]

        return sparse_border_mat, mat_pairs, border_pairs, border_times, is_event, states, delta_t, max_row_len

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
                self.get_x0_r().requires_grad = x0_grad

        # Set the gradient of the velocities
        if v_grad is not None:
            self.get_v_s(standardize=False).requires_grad = v_grad
            if self.is_directed():
                self.get_v_r().requires_grad = v_grad
                
        # Set the gradient of the all prior parameters
        if prior_grad is not None:
            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = prior_grad

    def learn(self, loss_file_path=None):

        # Check the learning procedure
        if self.__lp == "sequential":

            # Run sequential learning
            loss, nll = self.__sequential_learning()

        else:

            raise Exception(f"Unknown learning procedure: {self.__lp}")

        # Save the loss if the loss file path was given
        if loss_file_path is not None:
            
            with open(loss_file_path, 'w') as f:
                for batch_losses, nll_losses in zip(loss, nll):
                    f.write(f"Loss: {' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")
                    f.write(f"Nll: {' '.join('{:.3f}'.format(loss) for loss in nll_losses)}\n")

        return loss, nll

    def __sequential_learning(self):

        if self.get_verbose():
            print("- Training started (Procedure: Sequential Learning).")

        # Define the optimizers and parameter group names
        self.group_optimizers = []
        self.param_groups = [["v"], ["v", "x0"], ["v", "x0", "prior"]]
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
            torch.as_tensor([0] + self.group_epoch_weights, device=self.get_device(), dtype=torch.float), dim=0
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
                epoch_loss, epoch_nll = self.__train_one_epoch(epoch_num=epoch_num, optimizer=optimizer)
                loss.append(epoch_loss)
                nll.append(epoch_nll)

                # Increase the epoch number by one
                epoch_num += 1

        return loss, nll

    def __train_one_epoch(self, epoch_num, optimizer):

        init_time = time.time()

        total_batch_loss = 0
        epoch_loss, epoch_nll = [], []
        for batch_num in range(self.__steps_per_epoch):

            batch_nll, batch_nlp = self.__train_one_batch(batch_num)
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
            print(f"- Epoch loss is {avg_loss}, stopping training")
            sys.exit(1)

        if self.get_verbose() and (epoch_num % 10 == 0 or epoch_num == self.__epoch_num - 1):
            time_diff = time.time() - init_time
            print("\t+ Epoch = {} | Avg. Loss/train: {} | Elapsed time: {:.2f}".format(epoch_num, avg_loss, time_diff))

        return epoch_loss, epoch_nll

    def __train_one_batch(self, batch_num):

        self.train()

        batch_nodes = torch.multinomial(
            torch.ones(self.get_nodes_num(), dtype=torch.float, device=self.get_device()),
            self.__batch_size, replacement=False
        )
        batch_nodes, _ = torch.sort(batch_nodes, dim=0)

        # Construct the pairs of the batch
        batch_unique_pairs = torch.combinations(batch_nodes, r=2).T.type(torch.int)

        # Construct a sparse diagonal matrix of shape (N(N-1)/2 x N(N-1)/2), storing the sampled batch pairs
        idx = utils.pairIdx2flatIdx(batch_unique_pairs[0], batch_unique_pairs[1], self.get_nodes_num())
        sparse_batch_index_mat = torch.sparse_coo_tensor(
            torch.vstack((idx, idx)), values=torch.ones(size=(batch_unique_pairs.shape[1], ), dtype=torch.float, device=self.get_device()),
            size=(self.get_pairs_num(), self.get_pairs_num())
        )

        batch_pairs = utils.linearIdx2matIdx(
            torch.sparse.mm(sparse_batch_index_mat, self.__sparse_border_mat).coalesce().indices()[0], n=self.get_nodes_num()
        ).type(torch.long).T

        batch_borders = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__border_times, dtype=torch.float, device=self.get_device(),
                size=(self.get_pairs_num(), self.__max_row_len)
            )
        ).coalesce().values()

        batch_states = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__states, dtype=torch.float, device=self.get_device(),
                size=(self.get_pairs_num(), self.__max_row_len)
            )
        ).coalesce().type(torch.long).values()

        batch_is_event = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__is_event, dtype=torch.float, device=self.get_device(),
                size=(self.get_pairs_num(), self.__max_row_len)
            )
        ).coalesce().type(torch.int).values()

        batch_delta_t = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__delta_t, dtype=torch.float, device=self.get_device(),
                size=(self.get_pairs_num(), self.__max_row_len)
            )
        ).coalesce().values()

        # Compute the R factor inverse if the batch number is 0
        compute_R_factor_inv = True if batch_num == 0 else False

        # Finally, compute the negative log-likelihood and the negative log-prior for the batch
        average_batch_nll, average_batch_nlp = self.forward(
            nodes=batch_nodes, pairs=batch_pairs, states=batch_states, borders=batch_borders, 
            is_event=batch_is_event, delta_t=batch_delta_t, compute_R_factor_inv=compute_R_factor_inv
        )

        return average_batch_nll, average_batch_nlp

    def forward(self, nodes: torch.Tensor, pairs: torch.Tensor, states: torch.Tensor, borders: torch.Tensor, is_event=torch.Tensor, delta_t=torch.Tensor, compute_R_factor_inv=True):

        # Get the negative log-likelihood
        nll = self.get_nll(pairs=pairs, states=states, borders=borders, is_event=is_event, delta_t=delta_t)
        
        # Get the negative log-prior and the R-factor inverse
        nlp = self.get_neg_log_prior(nodes=nodes, compute_R_factor_inv=compute_R_factor_inv)

        # print(nll, nlp)

        return nll, nlp
        
    def save(self, path):

        if self.get_verbose():
            print(f"- Model file is saving.")
            print(f"\t+ Target path: {path}")

        # kwargs = {
        #     'data': [self.__events_pairs, self.__events ],
        #     'nodes_num': self.get_nodes_num(), 'bins_num': self.get_bins_num(), 'dim': self.get_dim(),
        #     'last_time': self.get_last_time(), 'approach': self.__approach,
        #     # 'prior_k': self.get_prior_k(), 'prior_lambda': self.get_prior_lambda(), 'masked_pairs': self.__masked_pairs,
        #     'learning_rate': self.__learning_rate, 'batch_size': self.__batch_size, 'epoch_num': self.__epoch_num,
        #     'steps_per_epoch': self.__steps_per_epoch,
        #     'device': self.get_device(), 'verbose': self.get_verbose(), 'seed': self.get_seed(),
        #     # 'learning_procedure': self.__learning_procedure,
        #     # 'learning_param_names': self.__learning_param_names,
        #     # 'learning_param_epoch_weights': self.__learning_param_epoch_weights
        # }

        kwargs = {
            'directed': self.is_directed(), 
            'init_states': self.get_init_states(),
            'prior_lambda': self.get_prior_lambda(), 
            'prior_R_factor_inv_s': self.get_prior_R_factor_inv_s(),
            'prior_R_factor_inv_r':  self.get_prior_R_factor_inv_r(),
            'device': self.get_device(), 'verbose': self.get_verbose(), 
            'seed': self.get_seed()
        }

        torch.save([kwargs, self.state_dict()], path)

        if self.get_verbose():
            print(f"\t+ Completed.")

