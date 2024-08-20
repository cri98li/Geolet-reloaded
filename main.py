import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from sklearn.cluster import KMeans, AffinityPropagation, OPTICS, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn_extra.cluster import KMedoids
from sklearn.tree import DecisionTreeClassifier

from geoletrld.model import Geolet
from geoletrld.selectors import RandomSelector, MutualInformationSelector, SelectorPipeline, ClusteringSelector, \
    GapSelector
from geoletrld.utils import Trajectories, y_from_df
from geoletrld.partitioners import NoPartitioner, GeohashPartitioner, FeaturePartitioner, SlidingWindowPartitioner
from geoletrld.distances import (EuclideanDistance, InterpolatedTimeDistance, LCSSTrajectoryDistance, FrechetDistance,
                                 CaGeoDistance, MatchComputeDistance, RotatingGenericDistance)


if __name__ == "__main__":
    print(FeaturePartitioner(feature='time', threshold=10))

    df = pd.read_csv('benchmark/datasets/vehicles.zip')
    #df = pd.read_csv("datasets/animals.zip")

    y = y_from_df(df, tid_name="tid", y_name="class")

    trajectories = Trajectories.from_DataFrame(df, latitude="c1", longitude="c2", time="t")

    X_train, X_test, y_train, y_test = train_test_split(list(trajectories), y, test_size=0.3, random_state=32,
                                                        stratify=y)
    X_train = Trajectories([(k, trajectories[k]) for k in X_train])
    X_test = Trajectories([(k, trajectories[k]) for k in X_test])

    classifier = Geolet(
        partitioner=GeohashPartitioner(precision=7),
        selector=MutualInformationSelector(n_jobs=8, k=5, distance=EuclideanDistance(), verbose=True),
        distance=MatchComputeDistance(EuclideanDistance(), CaGeoDistance()),
        #distance=EuclideanDistance(n_jobs=8),
        model_to_fit=RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=32, n_jobs=8),
        #model_to_fit=KMeans(n_clusters=2)
        subset_trj_in_selection=100,
        subset_candidate_geolet=100
    ).fit(X_train, y_train)

    print("Random Forest:\n", classification_report(y_test, classifier.predict(X_test)))
    #print(silhouette_score(classifier.transform(X_train), classifier.predict(X_train)))

    """geolets = NoPartitioner().transform(trajectories)
    print(geolets)

    geolets = GeohashPartitioner(precision=3).transform(trajectories)
    print(geolets)

    geolets = FeaturePartitioner(feature="distance", threshold=.2).transform(trajectories)
    print(geolets)

    geolets = SlidingWindowPartitioner(window_size=30).transform(trajectories)
    print(geolets)

    best_dist, best_idx = EuclideanDistance(agg=np.sum, n_jobs=8, verbose=True) \
        .transform(trajectories=trajectories, geolets=geolets)

    #print(best_dist)

    #best_dist, best_idx = InterpolatedTimeDistance(agg=np.sum, n_jobs=1, verbose=True) \
    #    .transform(trajectories=trajectories, geolets=geolets)

    print(best_dist)
    
    X_train, X_test, y_train, y_test = train_test_split(best_dist, y, test_size=0.3, random_state=42, stratify=y)

    dt = DecisionTreeClassifier().fit(X_train, y_train)
    print("Decision Tree:\n", classification_report(y_test, dt.predict(X_test)))

    rf = RandomForestClassifier().fit(X_train, y_train)
    print("Random Forest:\n", classification_report(y_test, rf.predict(X_test)))"""
