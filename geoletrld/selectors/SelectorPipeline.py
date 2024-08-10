import numpy as np

from geoletrld.selectors import SelectorInterface
from geoletrld.utils import Trajectories


class SelectorPipeline(SelectorInterface):

    def __init__(self, *pipeline):
        for arg in pipeline:
            if not isinstance(arg, SelectorInterface):
                raise ValueError('Argument must be an instance of SelectorInterface')

        self.pipeline = pipeline

    def select(self, geolets: Trajectories, trajectories: Trajectories = None, y: np.ndarray = None) -> (
            Trajectories, np.ndarray):
        selected_geolets = geolets
        scores = []
        for selector in self.pipeline:
            res = selector.select(geolets=selected_geolets, trajectories=trajectories, y=y)
            selected_geolets = res[0]
            scores.append(res[1])

        return selected_geolets, scores

    def __str__(self):
        s = []
        for el in self.pipeline:
            s.append(str(el)+", ")
        return f"Pipe({s}[:-2])"


