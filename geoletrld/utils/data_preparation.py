import numpy as np
import pandas as pd
from tqdm import tqdm


def y_from_df(df: pd.DataFrame, tid_name: str = "tid", y_name: str = "target", verbose: bool = False):
    y = []

    old_tid = None
    for tid, classe in tqdm(df[[tid_name, y_name]].values, disable=not verbose):
        if tid != old_tid:
            y.append(classe)
            old_tid = tid

    return np.array(y)
