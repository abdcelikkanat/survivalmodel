import numpy
import numpy as np
import random
import utils


class SurviveDieProcess:

    def __init__(self, lambda_func, initial_state: bool, critical_points: list, seed: int = 0):

        self.__lambda_func = lambda_func
        self.__state = initial_state
        self.__critical_points = critical_points
        self.__seed = seed

        assert self.__critical_points[0][0] == self.__critical_points[1][0], "FIX"
        self.__init_time = self.__critical_points[0][0]
        assert len(self.__critical_points[0]) == len(self.__critical_points[1]), "FIX"
        self.__num_of_bins = len(self.__critical_points[0])
        # Find the max lambda values for each interval; the first element is not used, just for indexing
        self.__lambda_max = ([0], [0])
        for idx in range(1, self.__num_of_bins):
            self.__lambda_max[0].append(
                max(
                    self.__lambda_func(t=self.__critical_points[0][idx-1], state=0),
                    self.__lambda_func(t=self.__critical_points[0][idx], state=0)
                )
            )
            self.__lambda_max[1].append(
                max(
                    self.__lambda_func(t=self.__critical_points[1][idx-1], state=1),
                    self.__lambda_func(t=self.__critical_points[1][idx], state=1)
                )
            )

        # Set seed
        random.seed(self.__seed)
        np.random.seed(self.__seed)

    def simulate(self) -> list:
        t, J, S = self.__init_time, 1, []
        # Step 2
        U = np.random.uniform(low=0, high=1)  # Random number
        X = (-1.0/(self.__lambda_max[1*self.__state][J]+utils.EPS)) * np.log(U)  # Random variable from exponential dist.

        while True:
            # Step 3
            if t + X < self.__critical_points[1*self.__state][J]:
                # Step 4
                t = t + X
                # Step 5
                U = np.random.uniform(low=0, high=1)
                # Step 6
                if U <= self.__lambda_func(t, state=self.__state)/self.__lambda_max[1*self.__state][J]:
                    # Don't need I for index, because we append t to S
                    S.append(t)
                    self.__state = (self.__state == False)
                # Step 7 -> Do step 2 then loop starts again at step 3
                U = np.random.uniform(low=0, high=1)  # Random number
                X = (-1./(self.__lambda_max[1*self.__state][J]+utils.EPS)) * np.log(U)  # Random variable from exponential dist.
            else:
                # Step 8
                if J == self.__num_of_bins - 1:  # k +1 because of zero-indexing
                    break
                # Step 9
                X = (X-self.__critical_points[1*self.__state][J] + t) * self.__lambda_max[1*self.__state][J]/self.__lambda_max[1*self.__state][J+1]
                t = self.__critical_points[1*self.__state][J]
                J += 1

        return S