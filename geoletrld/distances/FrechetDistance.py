from concurrent.futures import ProcessPoolExecutor

import numpy as np
from similaritymeasures import similaritymeasures
from tqdm.auto import tqdm

from geoletrld.distances.DistanceInterface import DistanceInterface
from geoletrld.utils import Trajectory, Trajectories


class FrechetDistance(DistanceInterface):
    def __init__(self, n_jobs=1, verbose=False):

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
            distances[i], best_idx[i] = FrechetDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize())

        return distances, best_idx

    @staticmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory,
            resample=False
    ) -> tuple:
        if resample:
            return FrechetDistance.__best_fitting_with_resample(trajectory, geolet)

        len_geo = len(geolet.latitude)
        len_trajectory = len(trajectory.latitude)

        if len_geo > len_trajectory:
            return .0, -1

        res = np.zeros(len_trajectory - len_geo + 1)
        for i in range(len_trajectory - len_geo + 1):
            trj_normalized, _ = Trajectory._first_point_normalize(trajectory.lat_lon[:, i:i + len_geo])
            res[i] = similaritymeasures.frechet_dist(trj_normalized, geolet.lat_lon)

        return min(res), np.argmin(res)

    @staticmethod
    def __best_fitting_with_resample(
            trajectory: Trajectory,
            geolet: Trajectory,
    ) -> tuple:
        time_delta = np.median(np.hstack([np.diff(trajectory.time), np.diff(geolet.time)]))
        geolet, _ = Trajectory._resample(geolet.time, geolet.lat_lon, f"{time_delta}s")
        trajectory, _ = Trajectory._resample(trajectory.time, trajectory.lat_lon, f"{time_delta}s")

        len_geo = geolet.shape[1]
        len_trajectory = trajectory.shape[1]

        if len_geo > len_trajectory:
            return .0, -1

        res = np.zeros(len_trajectory - len_geo + 1)
        for i in range(len_trajectory - len_geo + 1):
            trj_normalized, _ = Trajectory._first_point_normalize(trajectory[:, i:i + len_geo])
            res[i] = similaritymeasures.frechet_dist(trj_normalized, geolet)

        return min(res), np.argmin(res)

    def __str__(self):
        return f"Frechet({self.n_jobs}, {self.verbose})"
