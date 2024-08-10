from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from geoletrld.distances import DistanceInterface
from geoletrld.utils import Trajectory, Trajectories


class InterpolatedTimeDistance(DistanceInterface):
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
            distances[i], best_idx[i] = InterpolatedTimeDistance.best_fitting(
                trajectory=trajectory,
                geolet=geolet.normalize(),
                agg=self.agg)

        return distances, best_idx

    @staticmethod
    def best_fitting(trajectory: Trajectory, geolet: Trajectory, agg=np.sum) -> tuple[float, int]:
        len_geo = geolet.time[-1] - geolet.time[0]
        len_trajectory = trajectory.time[-1] - trajectory.time[0]

        if len_geo > len_trajectory:
            #return InterpolatedRouteDistance.best_fitting(geolet, trajectory, agg=agg)
            return .0, -1

        i_start = 0
        i_end = 1
        res = []
        res_t_start = []
        while i_end != len(trajectory.time):
            curr_trj_len = trajectory.time[i_end]-trajectory.time[i_start]
            if len_geo <= curr_trj_len:
                res.append(_ITD(trajectory.values[:, i_start:i_end], geolet.values, agg))
                res_t_start.append(i_start)
                i_start += 1
            else:
                i_end += 1

        return min(res), res_t_start[np.argmin(res)]

    def __str__(self):
        return f"ITD({self.agg.__name__}, {self.n_jobs}, {self.verbose})"


def _ITD(sub_trajectory: np.ndarray, geolets: np.ndarray, agg):
    sub_trajectory, _ = Trajectory._first_point_normalize(sub_trajectory)

    trj_lat_lon = sub_trajectory[:2]
    geo_lat_lon = geolets[:2]

    trj_time = sub_trajectory[2]
    geo_time = geolets[2]

    idx_trj = 0
    idx_geo = 0

    distances = []

    while True:
        if len(trj_time) == idx_trj or len(geo_time) == idx_geo:
            break

        if trj_time[idx_trj] == geo_time[idx_geo]:
            distances += [(trj_lat_lon[:, idx_trj] - geo_lat_lon[:, idx_geo]) ** 2]
            idx_trj += 1
            idx_geo += 1
        elif trj_time[idx_trj] > geo_time[idx_geo]:
            trj_lat_lon_tm1 = trj_lat_lon[:, :max(0, idx_trj - 1)+1]
            trj_time_tm1 = trj_time[:max(0, idx_trj - 1)+1]
            interpolated_lat_lon = linear_interpolation(trj_lat_lon_tm1,
                                                        trj_time_tm1,
                                                        trj_lat_lon[:, idx_trj],
                                                        trj_time[idx_trj],
                                                        geo_time[idx_geo])

            distances += [(geo_lat_lon[:, idx_geo] - interpolated_lat_lon) ** 2]
            idx_geo += 1
        elif trj_time[idx_trj] < geo_time[idx_geo]:
            idx_trj += 1

    return agg(distances)



def linear_interpolation(trj1: np.ndarray, t1: np.ndarray, trj2: np.ndarray, t2: np.float64, target_t: np.float64):
    trj1 = trj1[:, -1]
    t1 = t1[-1]

    if target_t < t1 or target_t > t2:
        raise ValueError("Target timestamp must be within the range [t1, t2]")

    interpolated_trj = trj1 + (trj2 - trj1) * ((target_t - t1) / (t2 - t1))
    return interpolated_trj


def linear_acceleration_interpolation(trj1: np.ndarray, t1: np.ndarray, trj2: np.ndarray, t2: float, target_t: float):
    if target_t < t1[-1] or target_t > t2:
        raise ValueError("Target timestamp must be within the range [t1, t2]")

    acc = _acceleration(np.hstack([trj1, trj2]), np.hstack([t1, t2]))[:, -2:]
    speed = _speed(np.hstack([trj1, trj2]), np.hstack([t1, t2]))[:, -2:]
    acc_lat = acc[0]
    acc_lon = acc[1]

    speed_lat = speed[0]
    speed_lon = speed[1]

    lat1 = trj1[0, -1] + speed_lat[-2]*(target_t-t1[-1])+.5*acc_lat[-1]*((target_t-t1[-1])**2)
    lon1 = trj1[1, -1] + speed_lon[-2]*(target_t-t1[-1])+.5*acc_lon[-1]*((target_t-t1[-1])**2)

    interpolated_trj = trj1[:, -1:] + (trj2 - trj1[:, -1:]) * ((target_t - t1[-1]) / (t2 - t1[-1]))
    return interpolated_trj


def _acceleration(trj: np.ndarray, time: np.ndarray):
    d_time = np.hstack([[0], time[1:] - time[:-1]])
    dist = np.hstack([np.array([[0, 0]]).T, trj[:, 1:] - trj[:, :-1]])

    return np.diff(dist / (d_time ** 2))


def _speed(trj: np.ndarray, time: np.ndarray):
    d_time = np.hstack([[0], time[1:] - time[:-1]])
    dist = np.hstack([np.array([[0, 0]]).T, trj[:, 1:] - trj[:, :-1]])

    return dist / d_time
