from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances.DistanceInterface import DistanceInterface
from geoletrld.utils import Trajectory, Trajectories


class MatchComputeDistance(DistanceInterface):
    def __init__(self, best_fitting_distance1, best_fitting_distance2, agg=np.sum, n_jobs=1, verbose=False):
        self.best_fitting_distance1 = best_fitting_distance1
        self.best_fitting_distance2 = best_fitting_distance2
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
            distances[i], best_idx[i] = MatchComputeDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize(),
                best_fitting_distance1=self.best_fitting_distance1,
                best_fitting_distance2=self.best_fitting_distance2,
                agg=self.agg)

        return distances, best_idx

    @staticmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory,
            best_fitting_distance1,
            best_fitting_distance2,
            agg=np.sum,
    ) -> tuple:
        len_geo = len(geolet.latitude)
        len_trajectory = len(trajectory.latitude)

        if len_geo > len_trajectory:
            #return EuclideanDistance.best_fitting(geolet, trajectory, agg=np.sum)
            return .0, -1

        _, idx = best_fitting_distance1(trajectory, geolet, agg=agg)

        sub_trj = Trajectory(values=trajectory.values[:, idx:idx + len_geo])

        dist, _ = best_fitting_distance2(sub_trj, geolet, agg=agg)

        if not np.isfinite(dist):
            print("HERE")

        dist, _ = best_fitting_distance2(sub_trj, geolet, agg=agg)

        return dist, idx
