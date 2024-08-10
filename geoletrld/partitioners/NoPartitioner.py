from typing import Any

import numpy as np

from geoletrld.partitioners.PartitionerInterface import PartitionerInterface
from geoletrld.utils.Trajectories import Trajectories
from geoletrld.utils.Trajectory import Trajectory


class NoPartitioner(PartitionerInterface):
    def transform(self, X: Trajectories[Any, Trajectory]) -> Trajectories[Any, Trajectory]:
        return_val = Trajectories()
        for i, (k, v) in enumerate(X.items()):
            return_val[f"{i}_{k}"] = v.copy()

        return return_val

    def __str__(self):
        return f"NoPartition()"
