from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_classif

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances import EuclideanDistance, DistanceInterface
from geoletrld.selectors.SelectorInterface import SelectorInterface
from geoletrld.selectors._SelectorUtils import compute_distance_selector
from geoletrld.utils import Trajectories, Trajectory


class MutualInformationSelector(SelectorInterface):
    def __init__(self, k: int | float = 10, distance: DistanceInterface = EuclideanDistance(), n_jobs: int = 1,
                 random_state: int = 42, verbose: bool = False):
        self.random_state = random_state
        self.k = k
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.distance = distance
        self.distance.n_jobs = n_jobs

    def select(self, geolets: Trajectories, trajectories: Trajectories = None, y: np.ndarray = None) -> \
            (Trajectories, np.ndarray):
        if trajectories is None or y is None:
            raise ValueError("A subset of trajectories relative target labels are required.")

        dist_matrix = compute_distance_selector(geolets=geolets, trajectories=trajectories, n_jobs=self.n_jobs,
                                                verbose=self.verbose, distance=self.distance)

        mi = mutual_info_classif(dist_matrix, y=y, random_state=self.random_state)

        top_k_positions = np.argpartition(mi, -self.k)[-self.k:]

        res = Trajectories()

        for i, (geolet_id, geolet) in enumerate(geolets.items()):
            if i in top_k_positions:
                res[geolet_id] = geolet

        return res, mi[top_k_positions]

    def __str__(self):
        return f"MI({self.k}, {self.distance}, {self.n_jobs}, {self.random_state}, {self.verbose})"
