from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances.DistanceInterface import DistanceInterface
from geoletrld.utils import Trajectory, Trajectories


class MatchComputeDistance(DistanceInterface):
    def __init__(self, distance1, distance2, n_jobs=1, verbose=False):
        self.distance1 = distance1
        self.distance1.n_jobs = n_jobs
        self.distance2 = distance2
        self.distance2.n_jobs = n_jobs

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
            distances[i], best_idx[i] = MatchComputeDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize(),
                distance1=self.distance1,
                distance2=self.distance2)

        return distances, best_idx

    @staticmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory,
            distance1,
            distance2,
    ) -> tuple:
        len_geo = len(geolet.latitude)
        len_trajectory = len(trajectory.latitude)

        if len_geo > len_trajectory:
            return .0, -1

        geolets = Trajectories()
        geolets["demo"] = geolet

        _, idx = distance1._compute_dist_geolets_trajectory(trajectory, geolets)

        sub_trj = Trajectory(values=trajectory.values[:, int(idx[0]):int(idx[0]) + len_geo])

        dist, _ = distance2._compute_dist_geolets_trajectory(sub_trj, geolets)

        if not np.isfinite(dist[0]):
            print("HERE")
            dist = [.0]

        return dist[0], int(idx[0])

    def __str__(self):
        return f"MatchCompute({self.distance1}, {self.distance2}, {self.n_jobs}, {self.verbose})"
