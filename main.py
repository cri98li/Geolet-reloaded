import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from geoletrld.utils import Trajectories, y_from_df
from geoletrld.partitioners import NoPartitioner, GeohashPartitioner, FeaturePartitioner, SlidingWindowPartitioner
from geoletrld.distances import EuclideanDistance

if __name__ == "__main__":
    df = pd.read_csv('datasets/animals.zip')

    y = y_from_df(df, tid_name="tid", y_name="class")

    trajectories = Trajectories.from_DataFrame(df, latitude="c1", longitude="c2", time="t")

    geolets = NoPartitioner().transform(trajectories)
    print(geolets)

    geolets = GeohashPartitioner(precision=3).transform(trajectories)
    print(geolets)

    geolets = FeaturePartitioner(feature="distance", threshold=.2).transform(trajectories)
    print(geolets)

    geolets = SlidingWindowPartitioner(window_size=10).transform(trajectories)
    print(geolets)

    best_dist, best_idx = EuclideanDistance(agg=np.sum, n_jobs=8, verbose=True)\
        .transform(trajectories=trajectories, geolets=geolets)

    print(best_dist)

    X_train, X_test, y_train, y_test = train_test_split(best_dist, y, test_size=0.3, random_state=42, stratify=y)

    dt = DecisionTreeClassifier().fit(X_train, y_train)
    print("Decision Tree:\n", classification_report(y_test, dt.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=500).fit(X_train, y_train)
    print("Decision Tree:\n", classification_report(y_test, rf.predict(X_test)))

