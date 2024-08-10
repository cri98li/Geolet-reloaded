import random

import numpy as np

from geoletrld.selectors.SelectorInterface import SelectorInterface
from geoletrld.utils import Trajectories


class RandomSelector(SelectorInterface):
    def __init__(self, k: int | float = 10, random_state: int = 42, verbose: bool = False):
        self.k = k
        self.verbose = verbose
        self.random_state = random_state
        random.seed(self.random_state)

    def select(self, geolets: Trajectories, trajectories: Trajectories = None, y: np.ndarray = None):
        selected_geolets_keys = random.sample(list(geolets.keys()), k=min(len(geolets), self.k))

        return dict([(k, geolets[k]) for k in selected_geolets_keys]), np.random.rand(self.k)

    def __str__(self):
        return f"Gap({self.k}, {self.random_state}, {self.verbose})"
