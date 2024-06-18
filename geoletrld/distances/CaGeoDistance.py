import warnings
from concurrent.futures import ProcessPoolExecutor
import random

import numpy as np
from similaritymeasures import similaritymeasures
from tqdm.auto import tqdm

from geoletrld.distances.DistanceInterface import DistanceInterface
from geoletrld.distances._DistancesUtils import cosine_distance
from geoletrld.utils import Trajectory, Trajectories
import CaGeo.algorithms.BasicFeatures as bf
import CaGeo.algorithms.SegmentFeatures as sf
import CaGeo.algorithms.AggregateFeatures as af

from TCIF.classes.T_CIF_observation import T_CIF_observations


class CaGeoDistance(DistanceInterface):
    def __init__(self, n_gaps=1, agg=np.sum, n_jobs=1, verbose=False, **kwargs):
        self.n_gaps = n_gaps
        self.agg = agg
        self.kwargs = kwargs

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
            distances[i], best_idx[i] = CaGeoDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize(),
                agg=self.agg)

        return distances, best_idx

    @staticmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory,
            agg=np.sum,
    ) -> tuple:
        len_geo = len(geolet.latitude)
        len_trajectory = len(trajectory.latitude)

        if len_geo > len_trajectory:
            # return EuclideanDistance.best_fitting(geolet, trajectory, agg=np.sum)
            return .0, -1

        geo_cageo = compute_feature(geolet.values, 1, None)

        res = np.zeros(len_trajectory - len_geo + 1)
        for i in range(len_trajectory - len_geo + 1):
            sub_trj_cageo = compute_feature(trajectory.values[:, i:i + len_geo])
            res[i] = agg(np.abs(sub_trj_cageo - geo_cageo))

        return min(res), np.argmin(res)


def compute_feature(trajectory, n_intervals=1, kwags=None):
    verbose = False

    if not verbose:
        warnings.filterwarnings('ignore')

    starts = [0]
    stops = [trajectory.shape[1]]

    if n_intervals != 1:
        starts = random.sample(range(0, trajectory.shape[1] - 3), n_intervals)
        stops = [start + random.randint(3, trajectory.shape[1] - start) for start in starts]

    feature = []
    it = tqdm(list(zip(starts, stops)), disable=not verbose, desc="Processing interval", leave=False, position=0)
    for start, stop in it:
        X_sub = trajectory[:, start:stop]

        it.set_description("Computing kinematic features")
        X_lat_sub, X_lon_sub, X_time_sub = X_sub

        dist = np.nan_to_num(bf.distance(X_lat_sub, X_lon_sub, accurate=False))

        transformed = [
            np.nan_to_num(bf.speed(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:])),
            dist,
            np.nan_to_num(bf.direction(X_lat_sub, X_lon_sub)),
            np.nan_to_num(bf.turningAngles(X_lat_sub, X_lon_sub)),
            np.nan_to_num(bf.acceleration(X_lat_sub, X_lon_sub, X_time_sub, accurate=dist[1:])),
        ]

        for arr in tqdm(transformed, disable=not verbose, desc="computing aggregate features", leave=False, position=1):
            for f in [af.sum, af.std, af.max, af.min, af.mean]:
                feature.append(f(arr, None))

        del transformed[:]
        feature.append([np.nan_to_num(sf.straightness(X_lat_sub, X_lon_sub))])
        feature.append([np.nan_to_num(sf.meanSquaredDisplacement(X_lat_sub, X_lon_sub))])
        feature.append([np.nan_to_num(sf.intensityUse(X_lat_sub, X_lon_sub))])
        feature.append([np.nan_to_num(sf.sinuosity(X_lat_sub, X_lon_sub))])

    return np.array(feature)
