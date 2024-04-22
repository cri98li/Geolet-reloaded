import numpy as np


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
        self.latitude = self.values[0]
        self.longitude = self.values[1]
        self.time = self.values[2]

    def copy(self):
        return Trajectory(values=self.values)
