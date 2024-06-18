import copy
from datetime import datetime

import numpy as np
import pandas as pd


class Trajectory:
    def __init__(self,
                 latitude: np.array = None,
                 longitude: np.array = None,
                 time: np.array = None,
                 values: np.array = None):
        if values is None:
            self.values = np.copy(np.array([latitude, longitude, time]))
        else:
            self.values = np.array(values)

        self.__set_values(self.values)

        self.normalized = None

        self.inverse_normalize_data = None

    def __set_values(self, values):
        self.values = values
        self.latitude = self.values[0]
        self.longitude = self.values[1]
        self.lat_lon = self.values[:2]
        self.time = self.values[2]

    def copy(self):
        copied_trj = Trajectory(values=self.values)
        copied_trj.normalized = self.normalized
        copied_trj.inverse_normalize_data = self.inverse_normalize_data
        return copied_trj

    @staticmethod
    def _first_point_normalize(values):
        values = np.copy(values)

        old_values = np.copy(values[:, 0])

        return values - values[:, 0].reshape(-1, 1), old_values

    @staticmethod
    def _first_point_inverse_normalize(values, old_values):
        if type(old_values) is not np.ndarray:
            old_values = np.array(old_values).reshape(-1, 1)

        return values + old_values

    @staticmethod
    def _resample(time: np.ndarray, values: np.ndarray, frequency: str = "30s", method: str = "linear"):
        cols_names = ["time_idx"] + [f"value_{i}" for i in range(values.shape[0])]
        df = pd.DataFrame(np.vstack([time, values]).T, columns=cols_names).set_index("time_idx")
        df.index = pd.to_datetime(list(map(lambda x: datetime.fromtimestamp(x), df.index)))
        df_resampled = df.resample(frequency).interpolate(method='linear').reset_index()
        df_resampled["index"] = df_resampled["index"].apply(lambda x: x.timestamp())

        return df_resampled.values[:, 1:].T, df_resampled.values[:, 0]

    def normalize(self, type='FirstPoint', inplace=True):
        if not inplace:
            return copy.deepcopy(self).normalize(inplace=True)

        if self.is_normalized():
            return self

        if type == 'FirstPoint':
            values, self.inverse_normalize_data = self._first_point_normalize(self.values)
            self.__set_values(values)
            self.normalized = 'FirstPoint'
            return self
        else:
            raise ValueError(f"Unknown normalization '{type}'")

    def inverse_normalize(self, inplace=True):
        if not self.is_normalized():
            raise ValueError(f"This trajectory has not been normalized")

        if not inplace:
            return Trajectory(values=self.values).inverse_normalize(inplace=True)

        if self.normalized == 'FirstPoint':
            values = self._first_point_inverse_normalize(self.values, self.inverse_normalize_data)
            self.__set_values(values)
            self.normalized = None
            self.inverse_normalize_data = None
            return self

    def is_normalized(self):
        return self.normalized is not None

    def head(self):
        head_trj = Trajectory(
            latitude=self.latitude[:-1],
            longitude=self.longitude[:-1],
            time=self.time[:-1],
        )
        head_trj.normalized = self.normalized
        head_trj.inverse_normalize_data = self.inverse_normalize_data

        return head_trj
