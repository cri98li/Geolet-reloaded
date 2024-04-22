from abc import ABC, abstractmethod
from typing import Any

from geoletrld.utils.Trajectories import Trajectories
from geoletrld.utils.Trajectory import Trajectory


class PartitionerInterface(ABC):
    @abstractmethod
    def transform(self, X: Trajectories[Any, Trajectory]) -> Trajectories[Any, Trajectory]:
        raise NotImplementedError()
