import numpy as np
import warnings
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from geoletrld.distances import EuclideanDistance, DistanceInterface
from geoletrld.selectors import SelectorInterface
from geoletrld.selectors._SelectorUtils import compute_symmetric_distance_selector
from geoletrld.utils import Trajectories


class ClusteringSelector(SelectorInterface):

    def __init__(self, clustering_fun=KMeans(n_clusters=10), distance: DistanceInterface = EuclideanDistance(),
                 use_sim=False, n_jobs: int = 1, verbose: bool = False):
        self.verbose = verbose
        self.clustering_fun = clustering_fun
        self.distance = distance
        self.distance.n_jobs = n_jobs
        self.use_sim = use_sim
        self.n_jobs = n_jobs

    def select(self, geolets: Trajectories, trajectories: Trajectories = None, y: np.ndarray = None) -> (
            Trajectories, np.ndarray):
        if trajectories is not None or y is not None:
            warnings.warn("Both the trajectories and y will not be used, present here for API consistency by "
                          "convention.")

        geolets_keys = np.array(list(geolets.keys()))

        dist_matrix = compute_symmetric_distance_selector(geolets=geolets, n_jobs=self.n_jobs, verbose=self.verbose,
                                                          distance=self.distance)

        if self.use_sim:
            dist_matrix *= -1

        self.labels = self.clustering_fun.fit_predict(dist_matrix)

        self.intra_cluster_distance = np.zeros((len(geolets),))

        for idx, label in enumerate(self.labels):
            self.intra_cluster_distance[idx] = np.sum(dist_matrix[idx][self.labels == label])

        res = Trajectories()
        scores = np.array(len(np.unique(self.labels)))

        for label in tqdm(np.unique(self.labels), desc="Collecting medoids", disable=not self.verbose):
            min_idx = np.argmin(self.intra_cluster_distance[self.labels == label])
            key = geolets_keys[self.labels == label][min_idx]
            res[f"cl={label}_{key}"] = geolets[key]
            scores = self.intra_cluster_distance[self.labels == label][min_idx]

        return res, scores

    def __str__(self):
        return f"Cluster({self.clustering_fun}, {self.distance}, {self.use_sim}, {self.n_jobs}, {self.verbose})"
