from itertools import product

import pandas as pd
import numpy as np
import psutil
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering

from sklearn_extra.cluster import KMedoids

import geoletrld.distances as dist
import geoletrld.partitioners as part
import geoletrld.selectors as sel
from geoletrld.distances._DistancesUtils import cosine_distance

def get_hyperparameters(n_jobs=-1):
    if n_jobs == -1:
        n_jobs = psutil.cpu_count(logical=False)

    distance_agg_fun = [np.sum, np.mean, np.std, np.min, np.max]

    distance_hyper = \
        [dist.EuclideanDistance(agg=x, n_jobs=n_jobs) for x in distance_agg_fun] + \
        [dist.CaGeoDistance(n_gaps=gaps, agg=agg, n_jobs=n_jobs) for gaps, agg in
         product([1, 3, 5], [np.sum, cosine_distance])] + \
        [dist.FrechetDistance(n_jobs=n_jobs)] + \
        [dist.InterpolatedTimeDistance(agg=agg, n_jobs=n_jobs) for agg in [np.sum, np.mean]] + \
        [dist.LCSSTrajectoryDistance(max_dist=1000, max_time=60, n_jobs=n_jobs) for d, t in product([1, 10, 100, 1000],
                                                                                                    [10, 30, 60])] + \
        [dist.RotatingGenericDistance(best_fitting_distance=d, n_jobs=n_jobs)
         for d in [dist.EuclideanDistance.best_fitting, dist.FrechetDistance.best_fitting,
                   dist.InterpolatedTimeDistance.best_fitting]] + \
        [dist.MatchComputeDistance(best_fitting_distance1=bf1, best_fitting_distance2=bf2) for bf1, bf2 in product(
            [
                dist.RotatingGenericDistance(best_fitting_distance=dist.EuclideanDistance(agg=np.sum), n_jobs=n_jobs),
                dist.RotatingGenericDistance(best_fitting_distance=dist.EuclideanDistance(agg=np.mean), n_jobs=n_jobs),
                dist.EuclideanDistance(n_jobs=n_jobs, agg=np.sum),
                dist.EuclideanDistance(n_jobs=n_jobs, agg=np.mean),
                dist.FrechetDistance(n_jobs=n_jobs),
            ],
            [
                dist.FrechetDistance(n_jobs=n_jobs),
                dist.LCSSTrajectoryDistance(max_dist=1000, max_time=60, n_jobs=n_jobs),
            ] + [dist.CaGeoDistance(n_gaps=gaps, agg=agg, n_jobs=n_jobs)
                 for gaps, agg in product([1, 3, 5], [np.sum, cosine_distance])]
        )]

    partition_hyper = \
        [part.GeohashPartitioner(precision=p) for p in range(2, 7+1)] + \
        [part.FeaturePartitioner(feature=f, threshold=t, overlapping=o)
         for f, t, o in product(
            ["time", "distance"],
            [10, 30, 60, 100, 500],
            [True, False]
        )] + \
        [part.FeaturePartitioner(feature=f, threshold=t, overlapping=o)
         for f, t, o in product(
            ["speed", "acceleration"],
            [5, 10, 50, 100],
            [True, False]
        )] + \
        [part.NoPartitioner()] + \
        [part.SlidingWindowPartitioner(window_size=s, overlap=o) for s, o in product([5, 10, 50, 100], [True, False])]

    n_geolets = [2, 5, 10, 50, 100]
    select_hyper_unsup = \
        [sel.ClusteringSelector(clustering_fun=KMeans(n_clusters=k, n_init=100)) for k in n_geolets] + \
        [sel.ClusteringSelector(clustering_fun=KMedoids(n_clusters=k, metric="precomputed")) for k in n_geolets] + \
        [sel.GapSelector(k=k) for k in n_geolets] + \
        [sel.RandomSelector(k=k) for k in n_geolets] + \
        [sel.ClusteringSelector(clustering_fun=AffinityPropagation(affinity="precomputed"), use_sim=True)] + \
        [sel.ClusteringSelector(clustering_fun=SpectralClustering(affinity="precomputed"), use_sim=True)]

    select_hyper_sup = [sel.MutualInformationSelector(k=k) for k in n_geolets]

    return {
        'clf': {
            "partitioner": partition_hyper,
            "selector": select_hyper_unsup+select_hyper_sup,
            "distance": distance_hyper,
            "subset_candidate_geolet": [100, 500, 1000, 5000],
            "subset_trj_in_selection": [50, 100, 500, 1000]
        },
        'clu': {
            "partitioner": partition_hyper,
            "selector": select_hyper_unsup,
            "distance": distance_hyper,
            "subset_candidate_geolet": [100, 500, 1000, 5000],
            "subset_trj_in_selection": [50, 100, 500, 1000]
        }
    }

