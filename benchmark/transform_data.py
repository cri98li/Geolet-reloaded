import argparse
import multiprocessing
import os
import pickle
import sys
import time
from itertools import product

n_workers_per_test = 8
os.environ["OMP_NUM_THREADS"] = f"{n_workers_per_test}"

import numpy as np
import pandas as pd
import psutil
from progress_table import ProgressTable
from sklearn.model_selection import StratifiedKFold, train_test_split

from benchmark.benchmark_utils import BoundedQueueProcessPoolExecutor
from benchmark.hyperparameters import get_hyperparameters
from geoletrld.model import Geolet
from geoletrld.utils import Trajectories, y_from_df
import traceback
import warnings
warnings.filterwarnings("ignore")



datasets = ["datasets/vehicles.zip", "datasets/animals.zip", "datasets/seabirds.zip"]


def fix_hyper(hyper_dict):
    distance = hyper_dict["distance"]
    selector = hyper_dict["selector"]
    selector.distance = distance
    selector.distance.n_jobs = n_workers_per_test

    return hyper_dict


def run_clu(hyper_set, trajectories):
    hyper_dict = dict(zip(get_hyperparameters(-1)['clu'].keys(), hyper_set))

    model = Geolet(**fix_hyper(hyper_dict))
    res = dict()
    res.update(hyper_dict)

    try:
        start = time.time()
        arr = model.fit_transform(trajectories)
        stop = time.time()

        res.update({
            "time": stop - start
        })

    except Exception as e:
        print(e)
        return res, model, None

    return res, model, arr

def eval_clu(future, table, filename, hyper_set, semaphore, dataset_name):
    semaphore.acquire()
    hyper_dict = dict(zip(get_hyperparameters(-1)['clu'].keys(), hyper_set))
    table["dataset"] = dataset_name
    table.update_from_dict(hyper_dict)
    res, model, X_transformed = future.result()
    res.update({"dataset": dataset_name})

    if "time" not in res:
        table["time"] = "error"
    else:
        table["time"] = res["time"]
        df = pd.DataFrame.from_dict([res])
        df.to_csv(filename, index=False)
        np.save(filename.replace(".csv", ".npy"), X_transformed)
        pickle.dump(model, open(filename.replace(".csv", ".pickle"), "wb"))
    table.next_row()
    semaphore.release()

def run_clf(hyper_set, X_train, y_train, X_test, y_test):
    hyper_dict = dict(zip(get_hyperparameters(-1)['clu'].keys(), hyper_set))

    model = Geolet(**fix_hyper(hyper_dict))
    res = dict()
    res.update(hyper_dict)

    try:
        start = time.time()
        X_train_transf = model.fit_transform(X_train, y_train)
        stop = time.time()
        X_test_transf = model.transform(X_test)

        res.update({
            "time": stop - start
        })
    except Exception as e:
        print(e)
        return res, None, None, None

    return res, model, X_train_transf, X_test_transf

def eval_clf(future, table, filename, hyper_set, cv_idx, semaphore, dataset_name):
    semaphore.acquire()
    hyper_dict = dict(zip(get_hyperparameters(-1)['clu'].keys(), hyper_set))
    table["dataset"] = dataset_name
    table["cv_idx"] = cv_idx
    table.update_from_dict(hyper_dict)
    res, model, X_train_transformed, X_test_transformed = future.result()
    res.update({"dataset": dataset_name, "cv_idx": cv_idx})

    if "time" not in res:
        table["time"] = "error"
    else:
        table["time"] = res["time"]
        df = pd.DataFrame.from_dict([res])
        df.to_csv(filename, index=False)
        np.save(filename.replace(".csv", "_train.npy"), X_train_transformed)
        np.save(filename.replace(".csv", "_test.npy"), X_test_transformed)
        pickle.dump(model, open(filename.replace(".csv", ".pickle"), "wb"))
    table.next_row()
    semaphore.release()


def main(MODE):
    hyper = get_hyperparameters(n_workers_per_test)[MODE]

    if MODE == 'clu':
        for dataset_path in datasets:
            dataset_name = dataset_path.split("/")[-1]

            df = pd.read_csv(dataset_path)
            trajectories = Trajectories.from_DataFrame(df, latitude="c1", longitude="c2", time="t")

            hyper_to_test = list(product(*hyper.values()))

            print(f"n_parallel test: {psutil.cpu_count(logical=False) // n_workers_per_test} "
                  f"each using {n_workers_per_test} cores",)
            print(f"\r\n{dataset_name}\r\n")
            table = ProgressTable(pbar_embedded=False,
                                  pbar_show_progress=True,
                                  pbar_show_percents=True,
                                  pbar_show_throughput=False,
                                  num_decimal_places=3)

            semaphore = multiprocessing.Semaphore(1)
            with BoundedQueueProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False) // n_workers_per_test,
                                                 max_waiting_tasks=psutil.cpu_count(logical=False) // n_workers_per_test // 2) as exe:
                for hyper_set in table(hyper_to_test):
                    hyper_set_str = "_".join([str(el) for el in hyper_set])
                    filename = f"{MODE}_{dataset_name}_{hyper_set_str}"
                    filename_hash = hash(filename)
                    path = f"transformations/{MODE}/{filename_hash}.csv"
                    if os.path.exists(path):
                        semaphore.acquire()
                        hyper_dict = dict(zip(hyper.keys(), hyper_set))
                        table["dataset"] = dataset_name
                        table.update_from_dict(hyper_dict)
                        table["time"] = "skip"
                        table.next_row()
                        semaphore.release()
                        continue
                    # (hyper_set, X, labels_dict:dict=None)
                    future = exe.submit(run_clu, hyper_set, trajectories)
                    future.add_done_callback(lambda x: eval_clu(x, table, path, hyper_set, semaphore, dataset_name))
            table.close()
            del table
            print()

    elif MODE == 'clf':
        for dataset_path in datasets:
            df = pd.read_csv(dataset_path)
            dataset_name = dataset_path.split("/")[-1].split(".")[0]

            y = y_from_df(df, tid_name="tid", y_name="class")
            trajectories = Trajectories.from_DataFrame(df, latitude="c1", longitude="c2", time="t")

            X_train, X_test, y_train, y_test = train_test_split(list(trajectories.items()), y, test_size=0.2,
                                                                random_state=42, stratify=y)

            X_train = Trajectories(X_train)
            X_test = Trajectories(X_test)  # una volta selezionati i modelli

            skf = StratifiedKFold(n_splits=5)
            for i, (train_index, test_index) in enumerate(skf.split(list(X_train.items()), y_train)):
                X_train_cv = [t for i, t in enumerate(X_train.items()) if i in train_index]
                y_train_cv = [_y for i, _y in enumerate(y_train) if i in train_index]
                X_val_cv = [t for i, t in enumerate(X_train.items()) if i in test_index]
                y_val_cv = [_y for i, _y in enumerate(y_train) if i in test_index]

                X_train_cv = Trajectories(X_train_cv)
                X_val_cv = Trajectories(X_val_cv)

                hyper_to_test = list(product(*hyper.values()))

                print(f"n_parallel test: {psutil.cpu_count(logical=False) // n_workers_per_test} "
                      f"each using {n_workers_per_test} cores", )
                print(f"\r\n{dataset_name}\r\n")
                table = ProgressTable(pbar_embedded=False,
                                      pbar_show_progress=True,
                                      pbar_show_percents=True,
                                      pbar_show_throughput=False,
                                      num_decimal_places=3)
                semaphore = multiprocessing.Semaphore(1)
                with BoundedQueueProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False) // n_workers_per_test,
                                                     max_waiting_tasks=psutil.cpu_count(
                                                         logical=False) // n_workers_per_test // 2) as exe:
                    for hyper_set in table(hyper_to_test):
                        hyper_set_str = "_".join([str(el) for el in hyper_set])
                        filename = f"{MODE}_{dataset_name}_{hyper_set_str}_{i}"
                        filename_hash = hash(filename)
                        path = f"transformations/{MODE}/{filename_hash}.csv"
                        if os.path.exists(path):
                            semaphore.acquire()
                            hyper_dict = dict(zip(hyper.keys(), hyper_set))
                            table.update_from_dict(hyper_dict)
                            table["cv_idx"] = i
                            table["accuracy"] = "skip"
                            table["macro_f1"] = "skip"
                            table.next_row()
                            semaphore.release()
                            continue

                        future = exe.submit(run_clf, hyper_set, X_train_cv, y_train_cv, X_val_cv, y_val_cv)
                        future.add_done_callback(lambda x: eval_clf(x, table, path, hyper_set, i, semaphore,
                                                                    dataset_name))
                table.close()
                del table
                print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the benchmark type ['clu', 'clf']")

    arg = sys.argv[1]
    if arg == "clu":
        print("Clustering benchmark")
    elif arg == "clf":
        print("Classification benchmark")
    else:
        raise Exception("Unsupported argument")
    main(arg)