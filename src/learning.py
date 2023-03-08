import sys
import math
import torch
from src.base import BaseModel
import time
import utils


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data, nodes_num, bins_num, dim, directed,
                 lr: float = 0.1, batch_size: int = None, epoch_num: int = 100, steps_per_epoch=10, 
                 device: torch.device = None, verbose: bool = False, seed: int = 19):

        super(LearningModel, self).__init__(
            x0=(
                torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False), 
                torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False) if directed else None, 
            ),
            v=(
                torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
                torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False) if directed else None,
            ),
            beta=(
                torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, 2), device=device), requires_grad=False),
                torch.nn.Parameter(2 * torch.zeros(size=(nodes_num, 2), device=device), requires_grad=False) if directed else None,
            ),
            prior_lambda=1.0,
            prior_sigma=1.0,
            prior_B_x0_c=1.0,
            prior_B_ls=1.0,
            prior_C_Q=1.0,
            init_states=torch.nn.Parameter(torch.zeros(size=((nodes_num-1)*nodes_num//2, ), device=device), requires_grad=False),
            bins_num=bins_num,
            directed=directed,
            device=device,
            verbose=verbose,
            seed=seed
        )

        # Optimization parameters
        self.__lp = "seq" # Learning procedure
        self.__lr = lr # Learning rate
        self.__epoch_num = epoch_num # Number of epochs
        self.__batch_size = nodes_num if batch_size is None else batch_size # Batch size
        self.__steps_per_epoch = steps_per_epoch # Number of steps per epoch
        self.__optimizer = self.torch.optim.Adam # Optimizer

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
            self.get_bin_bounds()[:-1].unsqueeze(0).expand(self.number_of_pairs(), self.get_bins_num()).flatten(),
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

        # print(border_times)
        # print(border_pairs)
        # print(is_event)

        #### FIX - FIX - FIX ####
        mat_row_indices = utils.pairIdx2flatIdx(border_pairs[0], border_pairs[1], n=self.get_nodes_num()).type(torch.long)
        counts = [0]*self.number_of_pairs() #torch.zeros(self.number_of_pairs(), dtype=torch.long, device=self.get_device())
        mat_col_indices = []
        for r in mat_row_indices:
            mat_col_indices.append(counts[r])
            counts[r] += 1
        mat_col_indices = torch.as_tensor(mat_col_indices, dtype=torch.long, device=self.get_device())
        max_row_len = max(counts) + 1
        mat_pairs = torch.vstack((mat_row_indices, mat_col_indices))
        #### FIX - FIX - FIX ####

        # Find the corresponding border times of the chosen times
        sparse_border_mat = torch.sparse_coo_tensor(mat_pairs, values=border_times, size=(self.number_of_pairs(), max_row_len))

        states = torch.sparse.mm(
            torch.sparse_coo_tensor(mat_pairs, values=is_event.type(torch.int), size=(self.number_of_pairs(), max_row_len)), 
            torch.triu(torch.ones(size=(max_row_len, max_row_len), dtype=torch.int, device=self.get_device()), diagonal=0)
        )[mat_row_indices, mat_col_indices]
        states[states % 2 == 0] = 0
        states[states > 0] = 1

        delta_t = torch.sparse.mm(
            torch.sparse_coo_tensor(mat_pairs, values=border_times, size=(self.number_of_pairs(), max_row_len)), 
            -torch.diag(torch.ones(max_row_len, dtype=torch.float, device=self.get_device()), diagonal=0) + 
            torch.diag(torch.ones(max_row_len-1, dtype=torch.float, device=self.get_device()), diagonal=-1) 
        )[mat_row_indices, mat_col_indices]
        delta_t[delta_t < 0] = self.get_last_time() + delta_t[delta_t < 0]

        # raise Exception("STOP")

        return sparse_border_mat, mat_pairs, border_pairs, border_times, is_event, states, delta_t, max_row_len
    # def __initialize(self, data):
    #     '''
    #     Compute the model variables

    #     :param data: A tuple of (pairs, times, states)
    #     '''

    #     pairs, times, states = data

    #     pairs = torch.as_tensor(pairs, dtype=torch.long, device=self.get_device()).T
    #     times = torch.as_tensor(times, dtype=torch.float, device=self.get_device())

    #     # Add the bin times for each pair and reconstruct the pairs and times tensors
    #     border_pairs = torch.hstack((
    #         torch.repeat_interleave(torch.triu_indices(
    #             self.get_number_of_nodes(), self.get_number_of_nodes(), offset=1, dtype=torch.long, device=self.get_device()
    #         ), repeats=self.get_bins_num(), dim=1),
    #         pairs
    #     ))
    #     border_times = torch.hstack((
    #         self.get_bins_bounds()[:-1].unsqueeze(0).expand(self.number_of_pairs(), self.get_bins_num()).flatten(),
    #         times
    #     ))

    #     # Construct a tensor indicating if the corresponding time is an event/link or not
    #     is_event = torch.hstack((
    #         torch.zeros(len(border_times)-len(times),  dtype=torch.bool, device=self.get_device()),
    #         torch.ones_like(times, dtype=torch.bool, device=self.get_device())
    #     ))

    #     # Sort all pairs, times and is_event vector
    #     sorted_indices = torch.argsort(border_times)
    #     border_pairs = border_pairs[:, sorted_indices]
    #     border_times = border_times[sorted_indices]
    #     is_event = is_event[sorted_indices]

    #     #### FIX - FIX - FIX ####
    #     mat_row_indices = utils.pairIdx2flatIdx(border_pairs[0], border_pairs[1], n=self.get_number_of_nodes()).type(torch.long)
    #     counts = [0]*self.number_of_pairs() #torch.zeros(self.number_of_pairs(), dtype=torch.long, device=self.get_device())
    #     mat_col_indices = []
    #     for r in mat_row_indices:
    #         mat_col_indices.append(counts[r])
    #         counts[r] += 1
    #     mat_col_indices = torch.as_tensor(mat_col_indices, dtype=torch.long, device=self.get_device())
    #     max_row_len = max(counts) + 1
    #     mat_pairs = torch.vstack((mat_row_indices, mat_col_indices))
    #     #### FIX - FIX - FIX ####

    #     # Find the corresponding border times of the chosen times
    #     sparse_border_mat = torch.sparse_coo_tensor(mat_pairs, values=border_times, size=(self.number_of_pairs(), max_row_len))

    #     states = torch.sparse.mm(
    #         torch.sparse_coo_tensor(mat_pairs, values=is_event.type(torch.int), size=(self.number_of_pairs(), max_row_len)), 
    #         torch.triu(torch.ones(size=(max_row_len, max_row_len), dtype=torch.int, device=self.get_device()), diagonal=0)
    #     )[mat_row_indices, mat_col_indices]
    #     states[states % 2 == 0] = 0
    #     states[states > 0] = 1

    #     delta_t = torch.sparse.mm(
    #         torch.sparse_coo_tensor(mat_pairs, values=border_times, size=(self.number_of_pairs(), max_row_len)), 
    #         -torch.diag(torch.ones(max_row_len, dtype=torch.float, device=self.get_device()), diagonal=0) + 
    #         torch.diag(torch.ones(max_row_len-1, dtype=torch.float, device=self.get_device()), diagonal=-1) 
    #     )[mat_row_indices, mat_col_indices]
    #     delta_t[delta_t < 0] = self.get_last_time() + delta_t[delta_t < 0]

    #     return sparse_border_mat, mat_pairs, border_pairs, border_times, is_event, states, delta_t, max_row_len

    def learn(self, loss_file_path=None):

        # Initialize optimizer list
        self.__optimizer = []

        # For each parameter group, add an optimizer
        for param_group in self.__learning_param_names:
            
            # Set the gradients to True
            for param_name in param_group:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # Add a new optimizer
            self.__optimizer.append(
                torch.optim.Adam(self.parameters(), lr=self.__lr)
            )
            # Set the gradients to False
            for param_name in param_group:
                self.__set_gradients(**{f"{param_name}_grad": False})

        # Run sequential learning
        self.__sequential_learning()

        # Save the loss if the loss file path was given
        if loss_file_path is not None:
            
            with open(loss_file_path, 'w') as f:
                for batch_losses, nll_losses in zip(self.__loss, self.__nll):
                    f.write(f"Loss: {' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")
                    f.write(f"Nll: {' '.join('{:.3f}'.format(loss) for loss in nll_losses)}\n")

    def __sequential_learning(self):

        if self.get_verbose():
            print("- Training started (Sequential Learning).")

        current_epoch = 0
        current_param_group_idx = 0
        group_epoch_counts = (self.__epoch_num * torch.cumsum(
            torch.as_tensor([0] + self.__learning_param_epoch_weights, device=self.get_device(), dtype=torch.float), dim=0
        ) / sum(self.__learning_param_epoch_weights)).type(torch.int)
        group_epoch_counts = group_epoch_counts[1:] - group_epoch_counts[:-1]

        while current_epoch < self.__epoch_num:

            # Set the gradients to True
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # Repeat the optimization of the group parameters given weight times
            for _ in range(group_epoch_counts[current_param_group_idx]):
                self.__train_one_epoch(
                    epoch_num=current_epoch, optimizer=self.__optimizer[current_param_group_idx]
                )
                current_epoch += 1

            # Iterate the parameter group id
            current_param_group_idx += 1

    def __train_one_epoch(self, epoch_num, optimizer):

        init_time = time.time()

        total_batch_loss = 0
        self.__loss.append([])
        self.__nll.append([])
        for batch_num in range(self.__steps_per_epoch):

            batch_loss, batch_nll = self.__train_one_batch(batch_num)

            self.__loss[-1].append(batch_loss)
            self.__nll[-1].append(batch_nll)

            total_batch_loss += batch_loss

        # Set the gradients to 0
        optimizer.zero_grad()

        # Backward pass
        total_batch_loss.backward()

        # Perform a step
        optimizer.step()

        # Get the average epoch loss
        epoch_loss = total_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(epoch_loss):
            print(f"- Epoch loss is {epoch_loss}, stopping training")
            sys.exit(1)

        if self.get_verbose() and (epoch_num % 10 == 0 or epoch_num == self.__epoch_num - 1):
            time_diff = time.time() - init_time
            print("\t+ Epoch = {} | Loss/train: {} | Elapsed time: {:.2f}".format(epoch_num, epoch_loss, time_diff))

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
            size=(self.number_of_pairs(), self.number_of_pairs())
        )

        batch_pairs = utils.linearIdx2matIdx(
            torch.sparse.mm(sparse_batch_index_mat, self.__sparse_border_mat).coalesce().indices()[0], n=self.get_nodes_num()
        ).type(torch.long).T

        batch_borders = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__border_times, dtype=torch.float, device=self.get_device(),
                size=(self.number_of_pairs(), self.__max_row_len)
            )
        ).coalesce().values()

        batch_states = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__states, dtype=torch.float, device=self.get_device(),
                size=(self.number_of_pairs(), self.__max_row_len)
            )
        ).coalesce().type(torch.long).values()

        batch_is_event = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__is_event, dtype=torch.float, device=self.get_device(),
                size=(self.number_of_pairs(), self.__max_row_len)
            )
        ).coalesce().type(torch.int).values()

        batch_delta_t = torch.sparse.mm(
            sparse_batch_index_mat, torch.sparse_coo_tensor(
                self.__mat_pairs, values=self.__delta_t, dtype=torch.float, device=self.get_device(),
                size=(self.number_of_pairs(), self.__max_row_len)
            )
        ).coalesce().values()

        average_batch_loss, average_batch_nll = self.forward(
            pairs=batch_pairs, states=batch_states, borders=batch_borders, is_event=batch_is_event, delta_t=batch_delta_t
        )

        return average_batch_loss, average_batch_nll


    def forward(self, pairs: torch.Tensor, states: torch.Tensor, borders: torch.Tensor, is_event=torch.Tensor, delta_t=torch.Tensor ):

        total, nll = 0, 0
        nll = self.get_nll(
            pairs=pairs, states=states, borders=borders, is_event=is_event, delta_t=delta_t
        )

        total = nll

        return total, nll

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, reg_params_grad=None):

        if beta_grad is not None:
            self.get_beta_s().requires_grad = beta_grad
            if self.is_directed():
                self.get_beta_r().requires_grad = beta_grad

        if x0_grad is not None:
            self.get_x0_s(standardize=False).requires_grad = x0_grad
            if self.is_directed():
                self.get_x0_r().requires_grad = x0_grad

        if v_grad is not None:
            self.get_v_s(standardize=False).requires_grad = v_grad
            if self.is_directed():
                self.get_v_r().requires_grad = v_grad

        if reg_params_grad is not None:

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = reg_params_grad

    def number_of_pairs(self):

        return (self.get_nodes_num()-1) * self.get_nodes_num() // 2

    def save(self, path):

        if self.get_verbose():
            print(f"- Model file is saving.")
            print(f"\t+ Target path: {path}")

        kwargs = {
            'data': [self.__events_pairs, self.__events ],
            'nodes_num': self.get_nodes_num(), 'bins_num': self.get_bins_num(), 'dim': self.get_dim(),
            'last_time': self.get_last_time(), 'approach': self.__approach,
            # 'prior_k': self.get_prior_k(), 'prior_lambda': self.get_prior_lambda(), 'masked_pairs': self.__masked_pairs,
            'learning_rate': self.__learning_rate, 'batch_size': self.__batch_size, 'epoch_num': self.__epoch_num,
            'steps_per_epoch': self.__steps_per_epoch,
            'device': self.get_device(), 'verbose': self.get_verbose(), 'seed': self.get_seed(),
            # 'learning_procedure': self.__learning_procedure,
            # 'learning_param_names': self.__learning_param_names,
            # 'learning_param_epoch_weights': self.__learning_param_epoch_weights
        }

        torch.save([kwargs, self.state_dict()], path)

        if self.get_verbose():
            print(f"\t+ Completed.")

