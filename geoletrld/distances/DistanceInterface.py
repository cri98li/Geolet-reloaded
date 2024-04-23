from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from geoletrld.utils import Trajectory, Trajectories


class DistanceInterface(ABC):
    @abstractmethod
    def transform(self, trajectories: Trajectories, geolets: Trajectories) -> tuple:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def best_fitting(
            trajectory: Trajectory,
            geolet: Trajectory
    ) -> tuple[float, int]:
        raise NotImplementedError()
