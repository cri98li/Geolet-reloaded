from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances.DistanceInterface import DistanceInterface
from geoletrld.utils import Trajectory, Trajectories


class EuclideanDistance(DistanceInterface):
    def __init__(self, agg=np.sum, n_jobs=1, verbose=False):
        self.agg = agg

        self.n_jobs = n_jobs
        self.verbose = verbose

    def transform(self, trajectories: Trajectories, geolets: Trajectories) -> tuple:
        if self.n_jobs == 1:
            distances = np.zeros((len(trajectories), len(geolets)))
            best_idx = np.zeros((len(trajectories), len(geolets)), dtype=int)

            for i, (_, trajectory) in enumerate(tqdm(trajectories.items(), disable=not self.verbose)):
                distances[i], best_idx[i] = self._compute_dist_geolets_trajectory(trajectory, geolets)

            return distances, best_idx
        else:
            executor = ProcessPoolExecutor(max_workers=self.n_jobs)
            processes = []
            for _, trajectory in trajectories.items():
                processes += [
                    executor.submit(self._compute_dist_geolets_trajectory, trajectory, geolets)
                ]

            distances = np.zeros((len(trajectories), len(geolets)))
            best_idx = np.zeros((len(trajectories), len(geolets)), dtype=int)

            for i, process in enumerate(tqdm(processes, disable=not self.verbose)):
                distances[i], best_idx[i] = process.result()

            return distances, best_idx

    def _compute_dist_geolets_trajectory(self, trajectory: Trajectory, geolets: Trajectories):
        distances = np.zeros(len(geolets))
        best_idx = np.zeros(len(geolets))
        for i, (_, geolet) in enumerate(geolets.items()):
            distances[i], best_idx[i] = EuclideanDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize(),
                agg=self.agg)

        return distances, best_idx

    @staticmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory,
            agg=np.sum
    ) -> tuple:
        len_geo = len(geolet.latitude)
        len_trajectory = len(trajectory.latitude)

        if len_geo > len_trajectory:
            #return EuclideanDistance.best_fitting(geolet, trajectory, agg=np.sum)
            return .0, -1

        res = np.zeros(len_trajectory - len_geo + 1)
        for i in range(len_trajectory - len_geo + 1):
            trj_normalized, _ = Trajectory._first_point_normalize(trajectory.lat_lon[:, i:i + len_geo])
            res[i] = agg(((trj_normalized - geolet.lat_lon) ** 2))

        return min(res), np.argmin(res)

    def __str__(self):
        return f"Euclidean({self.agg.__name__}, {self.n_jobs}, {self.verbose})"
