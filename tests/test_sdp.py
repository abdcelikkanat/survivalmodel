import unittest
import torch
from src.sdp import SurviveDieProcess

SEED = 19

class TestSurviveDieProcess(unittest.TestCase):

    def test_constant(self):

        # Define the initial state
        initial_state = 0

        # Define the critical points
        critical_points = torch.as_tensor([0, 1., 2.,], dtype=torch.float)

        # Define a step function
        lambda_func = lambda t, state: 2

        sdp = SurviveDieProcess(lambda_func=lambda_func, initial_state=initial_state, critical_points=critical_points, seed=SEED)
        edge_times, edge_states = sdp.simulate()

        self.assertListEqual(
            edge_times, 
            [0.015950242057442665, 0.0799078419804573, 0.8458676338195801, 1.3543199300765991, 1.4353907108306885]
        )

        self.assertListEqual(
            edge_states, [1-(value % 2) for value in range(initial_state, initial_state+len(edge_times))]
        )

    def test_triangular(self):

        # Define the initial state
        initial_state = 0

        # Define the critical points
        critical_points = torch.as_tensor([0, 1., 2., 3., 4.], dtype=torch.float)

        # Define a step function
        lambda_func = lambda t, state: t if t < 2. else 4. - t 

        sdp = SurviveDieProcess(lambda_func=lambda_func, initial_state=initial_state, critical_points=critical_points, seed=SEED)
        edge_times, edge_states = sdp.simulate()

        self.assertListEqual(
            edge_times, 
            [1.3458672761917114, 1.8543195724487305, 1.9353903532028198, 2.632775068283081]
        )

        self.assertListEqual(
            edge_states, [1-(value % 2) for value in range(initial_state, initial_state+len(edge_times))]
        )


if __name__ == '__main__':
    unittest.main()