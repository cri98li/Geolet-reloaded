import numpy as np
import pandas as pd

from geoletrld.utils.Trajectory import Trajectory


class Trajectories(dict):

    def copy(self):
        trajectories = Trajectories()
        for k, v in self.items():
            trajectories[k] = v.copy()

        return trajectories
    def numpy(self, return_keys=False):
        max_len = max([len(x) for _, x in self.items()])
        return_value = np.ones((len(self), max_len)) * np.inf

        for i, (k, v) in enumerate(self.items()):
            return_value[i, :len(v)] = v

        if return_keys:
            return return_value, np.array(self.keys())
        return return_value

    @staticmethod
    def from_DataFrame(df: pd.DataFrame, tid='tid', latitude='latitude', longitude='longitude', time='time'):
        df = df.copy()
        trajectories = Trajectories()
        df.sort_values(by=[tid, time], inplace=True)

        for key in df[tid].unique():
            sub_df = df[df[tid] == key]
            trajectories[key] = Trajectory(sub_df[latitude], sub_df[longitude], sub_df[time])

        return trajectories

    def remove_short_trajectories(self, inplace=True):
        if not inplace:
            self.copy().remove_short_trajectories(inplace=True)

        keys = list(self.keys())
        for key in keys:
            if len(self[key].time) < 3:
                self.pop(key)

        return self
