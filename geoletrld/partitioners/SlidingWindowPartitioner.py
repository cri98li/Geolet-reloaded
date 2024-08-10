from typing import Any

from tqdm import tqdm

from geoletrld.partitioners import PartitionerInterface
from geoletrld.utils import Trajectories, Trajectory


class SlidingWindowPartitioner(PartitionerInterface):
    def __init__(self, window_size: int | float = 100, overlap: int | float = 1, verbose: bool = False):
        self.window_size = window_size
        if type(window_size) is int:
            self.overlap = (window_size * overlap) // 1
        self.verbose = verbose

    def transform(self, X: Trajectories[Any, Trajectory]) -> Trajectories[Any, Trajectory]:
        candidate_geolet = Trajectories[Any, Trajectory]()

        for k, v in tqdm(X.items(), disable=not self.verbose):
            latitude = v.latitude
            longitude = v.longitude
            time = v.time

            step_size = self.overlap
            window_size = self.window_size

            if type(self.window_size) is float:
                window_size = (len(time) * self.window_size) // 1
                step_size = (window_size * self.overlap) // 1

            for c, start_idx in enumerate(range(0, len(time) - window_size + 1, step_size)):
                candidate_geolet[f"{c}_{k}"] = Trajectory(latitude[start_idx:start_idx + window_size],
                                                          longitude[start_idx:start_idx + window_size],
                                                          time[start_idx:start_idx + window_size])

        return candidate_geolet

    def __str__(self):
        return f"Sliding({self.window_size}, {self.overlap}, {self.verbose})"
