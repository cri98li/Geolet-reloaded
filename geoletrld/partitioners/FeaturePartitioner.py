from typing import Any

import numpy as np
from tqdm.auto import tqdm

from geoletrld.partitioners.PartitionerInterface import PartitionerInterface
from geoletrld.utils.Trajectories import Trajectories
from geoletrld.utils.Trajectory import Trajectory
import CaGeo.algorithms.BasicFeatures as bf


class FeaturePartitioner(PartitionerInterface):
    def __init__(self, feature, threshold, overlapping=False, verbose=True):
        if feature not in ["time", "distance", "speed", "acceleration"]:
            raise ValueError(f"Feature must be in ['time', 'distance', 'speed']. {feature} unsupported")
        self.feature = feature
        self.threshold = threshold
        self.overlapping = overlapping
        self.verbose = verbose

    def transform(self, X: Trajectories[Any, Trajectory]) -> Trajectories[Any, Trajectory]:
        candidate_geolet = Trajectories[Any, Trajectory]()

        for k, trajectory in tqdm(X.items(), disable=not self.verbose):
            latitude = trajectory.latitude
            longitude = trajectory.longitude
            time = trajectory.time

            feat = []
            match self.feature:
                case "time":
                    feat = time
                case "distance":
                    feat = np.cumsum(bf.distance(latitude, longitude)*1000)
                case "speed":
                    feat = bf.speed(latitude, longitude, time) * 3600
                case "acceleration":
                    feat = bf.acceleration(latitude, longitude, time) * 3600

            old_j = 0
            c = 0
            j = 0
            while True:
                if j == len(feat):
                    break
                value = feat[j]

                if (self.feature in ["time", "distance"] and value - feat[old_j] > self.threshold) \
                        or (self.feature in ["speed", "acceleration"] and value > self.threshold):
                    candidate_geolet[f"{c}_{k}"] = Trajectory(latitude[old_j:j], longitude[old_j:j], time[old_j:j])

                    if self.overlapping:
                        old_j += 1
                        continue
                    else:
                        old_j = j
                    c += 1

                j += 1

        return candidate_geolet.remove_short_trajectories(inplace=True)
