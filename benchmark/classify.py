import time
from glob import glob
import os
from hashlib import md5
from itertools import product

import numpy as np
import pandas as pd
import psutil
from progress_table import ProgressTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from benchmark.transform_data import datasets
from geoletrld.utils import y_from_df, Trajectories

BASE_SAVE_PATH = "scores/clf/"

models = {
    "DT": DecisionTreeClassifier,
    "RF": RandomForestClassifier,
    "KNN": KNeighborsClassifier
}

hypers = {
    "DT": {
        'max_depth': [1, 2, 3, 5, 10, 20, None],
        'class_weight': [None, 'balanced']
    },
    "RF": {
        'n_estimators': [100, 200],
        'max_depth': [1, 2, 3, 5, 10, 20, None],
        'class_weight': [None, 'balanced'],
        'bootstrap': [True, False]
    },
    "KNN": {
        'n_neighbors': [1, 3, 5, 10],
        'weights': ['uniform', 'distance']
    }
}

def evaluate_clf(y_test, y_pred, y_pred_proba):
    res = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "micro_f1": f1_score(y_test, y_pred, average="micro"),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
            "micro_precision": precision_score(y_test, y_pred, average="micro"),
            "macro_precision": precision_score(y_test, y_pred, average="macro"),
            "weighted_precision": precision_score(y_test, y_pred, average="weighted"),
            "micro_recall": recall_score(y_test, y_pred, average="micro"),
            "macro_recall": recall_score(y_test, y_pred, average="macro"),
            "weighted_recall": recall_score(y_test, y_pred, average="weighted"),
        }

    return res


if __name__ == '__main__':
    datasets_y = dict()

    for dataset_path in tqdm(datasets[:1]):
        df = pd.read_csv(dataset_path)
        dataset_name = dataset_path.split("/")[-1].split(".")[0]

        y = np.array(y_from_df(df, tid_name="tid", y_name="class"))
        trajectories = Trajectories.from_DataFrame(df, latitude="c1", longitude="c2", time="t")

        X_train, _, y_train, y_test = train_test_split(list(trajectories.items()), y, test_size=0.2,
                                                            random_state=42, stratify=y)

        skf = StratifiedKFold(n_splits=5)

        for idx_vc, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            if idx_vc == 0:
                datasets_y[dataset_name] = ([], [], y_train, y_test)

            y_train_cv = y[train_index]
            y_val_cv = y[test_index]
            datasets_y[dataset_name][0].append(y_train_cv)
            datasets_y[dataset_name][1].append(y_val_cv)

    for exp_path in glob('transformations/*/*.csv'):
        exp_hash = exp_path.split("/")[-1].split("_")[0]
        for model_name, model_constructor in models.items():
            table = ProgressTable(pbar_embedded=False,
                                  pbar_show_progress=False,
                                  pbar_show_percents=False,
                                  pbar_show_throughput=True,
                                  num_decimal_places=3)

            for hyper_set in product(*hypers[model_name].values()):
                hyper_dict = {k: v for k, v in zip(hypers[model_name].keys(), hyper_set)}

                table.update_from_dict({"exp_hash": exp_hash, "model_name": model_name} | hyper_dict)

                hyper_set_str = "_".join([str(el) for el in hyper_set])
                filename = f"clf_{dataset_name}_{model_name}_{exp_hash}_{hyper_set_str}"
                filename_hash = md5(filename.encode()).hexdigest()

                if os.path.exists(BASE_SAVE_PATH + filename_hash + ".csv"):
                    table['output'] = "skip"
                    table.next_row()
                    continue

                X_train = np.load(exp_path.replace(".csv", "_train.npy"))
                X_test = np.load(exp_path.replace(".csv", "_test.npy"))
                df_res = pd.read_csv(exp_path)
                cv_idx = df_res.cv_idx.iloc[0]
                dataset = df_res.dataset.iloc[0]
                table["train shape"] = X_train.shape
                table["test shape"] = X_test.shape

                if X_train.shape[1] == 0:
                    table["output"] = "error"
                    table.next_row()
                    continue

                y_train = datasets_y[dataset_name][0][cv_idx]
                y_test = datasets_y[dataset_name][1][cv_idx]

                clf = model_constructor(**hyper_dict)

                start = time.time()
                clf.fit(X_train, y_train)
                end = time.time()
                df_res["clf_train_time"] = end - start
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)

                df_measures = pd.DataFrame.from_dict([evaluate_clf(y_pred=y_pred, y_test=y_test, y_pred_proba=y_pred_proba)])

                df_res[df_measures.columns] = df_measures

                df_res.to_csv(BASE_SAVE_PATH + filename + ".csv", index=False)

                table['output'] = df_measures['macro_f1'].iloc[0]
                table.next_row()

            table.close()




