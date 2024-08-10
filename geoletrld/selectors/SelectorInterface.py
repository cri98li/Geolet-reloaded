from abc import ABC, abstractmethod

import numpy as np

from geoletrld.utils import Trajectories


class SelectorInterface(ABC):
    @abstractmethod
    def select(self, geolets: Trajectories, trajectories: Trajectories = None, y: np.ndarray = None) \
            -> (Trajectories, np.ndarray):
        pass


