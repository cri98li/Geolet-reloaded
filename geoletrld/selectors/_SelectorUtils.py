from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from geoletrld.utils import Trajectories, Trajectory


def compute_distance_selector(geolets: Trajectories, trajectories: Trajectories = None, n_jobs: int = 1,
                              verbose: bool = False, distance=None):
    dist_matrix = np.zeros((len(trajectories), len(geolets)))

    if n_jobs > 1:
        executor = ProcessPoolExecutor(max_workers=n_jobs)
        processes = []

        for geolet in tqdm(geolets.values(), disable=not verbose, desc="Submitting task"):
            processes.append(executor.submit(_compute_distances_selector, distance, geolet, trajectories))

        for i, process in enumerate(tqdm(processes, disable=not verbose, desc="Retrieving task")):
            dist_matrix[:, i] = process.result()

        executor.shutdown(wait=False)
    else:
        for i, geolet in enumerate(tqdm(geolets.values(), disable=not verbose, desc="Computing distances")):
            dist_matrix[:, i] = _compute_distances_selector(distance, geolet, trajectories)

    return dist_matrix

def compute_symmetric_distance_selector(geolets: Trajectories, n_jobs: int = 1, verbose: bool = False, distance=None):
    geolets = geolets.copy()
    geolets_list = list(geolets.copy().items())
    dist_matrix = np.zeros((len(geolets), len(geolets)))

    if n_jobs > 1:
        executor = ProcessPoolExecutor(max_workers=n_jobs)
        processes = []

        for i, (geolet_id, geolet_value) in enumerate(tqdm(geolets_list[:-1], disable=not verbose,
                                                           desc="Submitting task")):
            geolets.pop(geolet_id)
            processes.append(executor.submit(_compute_distances_selector, distance, geolet_value, geolets.copy()))

        for i, process in enumerate(tqdm(processes, disable=not verbose, desc="Retrieving task")):
            res = process.result()
            dist_matrix[i, i + 1:] = res
            dist_matrix[i + 1:, i] = res

        executor.shutdown(wait=True)
    else:
        for i, (geolet_id, geolet_value) in enumerate(tqdm(geolets_list[:-1], disable=not verbose,
                                                           desc="Computing distances")):
            geolets.pop(geolet_id)
            res = _compute_distances_selector(distance, geolet_value, geolets)
            dist_matrix[i, i + 1:] = res
            dist_matrix[i + 1:, i] = res

    return dist_matrix

def _compute_distances_selector(distance, geolet: Trajectory, trajectories: Trajectories):
    distances = np.zeros(len(trajectories))
    geolets = Trajectories()
    geolets["demo"] = geolet

    for i, (trajectory_id, trajectory) in enumerate(trajectories.items()):
        distances[i], idx = distance._compute_dist_geolets_trajectory(trajectory, geolets)

    return distances

