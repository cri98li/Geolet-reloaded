import warnings
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_classif

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances import EuclideanDistance, DistanceInterface
from geoletrld.selectors.SelectorInterface import SelectorInterface
from geoletrld.selectors._SelectorUtils import compute_distance_selector
from geoletrld.utils import Trajectories, Trajectory


class GapSelector(SelectorInterface):
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
        if trajectories is None:
            raise ValueError("A subset of trajectories is required.")

        dist_matrix = compute_distance_selector(geolets=geolets, trajectories=trajectories, n_jobs=self.n_jobs,
                                                verbose=self.verbose, distance=self.distance)#geo x trj

        return self._select_loop(self.k, dist_matrix, geolets)


    @classmethod
    def _select_loop(cls, k, dist_matrix, geolets):
        selected_geolets_position = set()
        selected_geolets_gap_scores = []
        prec_dist_matrix_size = (None, None)

        while len(selected_geolets_position) < k:
            if prec_dist_matrix_size == dist_matrix.shape:
                break
            prec_dist_matrix_size = dist_matrix.shape
        #for i in range(self.k):
            gaps = np.zeros((dist_matrix.shape[1], ))
            gaps_value = np.zeros((dist_matrix.shape[1], ))
            for geo_idx in range(dist_matrix.shape[1]):
                gaps[geo_idx], gaps_value[geo_idx] = cls.__compute_gap(k, dist_matrix[:, geo_idx])

            if max(gaps) == -np.inf:
                warnings.warn(f"Early stopping with {len(selected_geolets_position)} geolet")
                break

            selected_geolets_position.add(np.argmax(gaps))
            selected_geolets_gap_scores.append(max(gaps))
            threshold_distance = gaps_value[np.argmax(gaps)]
            threshold = np.mean(
                dist_matrix[dist_matrix[:, np.argmax(gaps)] < threshold_distance, np.argmax(gaps)]
            ) + np.std(
                dist_matrix[dist_matrix[:, np.argmax(gaps)] < threshold_distance, np.argmax(gaps)]
            )
            rows_to_keep = dist_matrix[:, np.argmax(gaps)] >= threshold
            dist_matrix = np.delete(dist_matrix[rows_to_keep], np.argmax(gaps), axis=1)


        res = Trajectories()
        for i, (geolet_id, geolet) in enumerate(geolets.items()):
            if i in selected_geolets_position:
                res[geolet_id] = geolet

        return res, np.array(selected_geolets_gap_scores)

    @classmethod
    def __compute_gap(cls, k, dist_vector):
        ordered_distances = dist_vector[np.argsort(dist_vector)]
        len_distances = len(dist_vector)
        max_gap = -np.inf
        best_d = -1
        for i in range(len_distances-1):
            D_A = ordered_distances[:i+1]
            D_B = ordered_distances[i+1:]
            ratio = len(D_A)/len(D_B)
            if 1./k < ratio < 1-(1./k):
                mean_a = np.mean(D_A)
                mean_b = np.mean(D_B)
                std_a = np.std(D_A)
                std_b = np.std(D_B)
                gap = mean_b - std_b - (mean_a+std_a)
                if gap > max_gap:
                    max_gap = gap
                    best_d = (ordered_distances[i]+ordered_distances[i+1])/2

        return max_gap, best_d

    def __str__(self):
        return f"Gap({self.k}, {self.distance}, {self.n_jobs}, {self.random_state}, {self.verbose})"
