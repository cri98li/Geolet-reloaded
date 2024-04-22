import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from geoletrld.utils import Trajectories
from geoletrld.partitioners import NoPartitioner, GeohashPartitioner, FeaturePartitioner, SlidingWindowPartitioner

if __name__ == "__main__":
    df = pd.read_csv('datasets/animals.zip')

    res = Trajectories.from_DataFrame(df, latitude="c1", longitude="c2", time="t")

    res2 = NoPartitioner().transform(res)
    print(res2)

    res2 = GeohashPartitioner(precision=4).transform(res)
    print(res2)

    res2 = FeaturePartitioner(feature="distance", threshold=.2).transform(res)
    print(res2)

    res2 = SlidingWindowPartitioner(window_size=10).transform(res)
    print(res2)

    res2[list(res2.keys())[0]].normalize()