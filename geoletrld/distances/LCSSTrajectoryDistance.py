from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances.DistanceInterface import DistanceInterface
from geoletrld.distances._DistancesUtils import haversine
from geoletrld.utils import Trajectory, Trajectories


class LCSSTrajectoryDistance(DistanceInterface):
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
            distances[i], best_idx[i] = LCSSTrajectoryDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize(),
                agg=self.agg)

        return distances, best_idx

    @staticmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory,
            max_dist=1000,
            max_time=60,
            agg=None
    ) -> tuple:
        len_geo = len(geolet.latitude)
        len_trajectory = len(trajectory.latitude)
        if not trajectory.is_normalized():
            trajectory = trajectory.normalize(inplace=False)

        dist_matrix = np.zeros((len_geo+1, len_trajectory+1), dtype=float)
        len_matrix = np.zeros((len_geo+1, len_trajectory+1), dtype=int)
        for i in range(len_geo):
            for j in range(len_trajectory):
                geo_lat = geolet.latitude[i]
                geo_lon = geolet.longitude[i]
                trj_lat = trajectory.latitude[j]
                trj_lon = trajectory.longitude[j]
                dist = haversine(geo_lon, geo_lat, trj_lon, trj_lat)
                
                geo_time = geolet.time[i]
                trj_time = trajectory.time[j]

                if abs(geo_time-trj_time) <= max_time and dist <= max_dist:
                    dist_matrix[i+1, j+1] = dist/max_dist + dist_matrix[i, j]
                    len_matrix[i+1, j+1] = 1 + len_matrix[i, j]
                else:
                    if dist_matrix[i+1, j] > dist_matrix[i, j+1]:
                        dist_matrix[i+1, j+1] = dist_matrix[i+1, j]
                        len_matrix[i + 1, j + 1] = 1 + len_matrix[i+1, j]
                    else:
                        dist_matrix[i + 1, j + 1] = dist_matrix[i, j+1]
                        len_matrix[i + 1, j + 1] = 1 + len_matrix[i, j+1]

        return -dist_matrix[-1, -1]/len_matrix[-1, -1], -1
