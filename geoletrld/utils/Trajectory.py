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
        self.lat_lon = self.values[:2]
        self.time = self.values[2]

    def copy(self):
        return Trajectory(values=self.values)

    def _first_point_normalize(self, inplace=True):
        if not inplace:
            return Trajectory(values=self.values)._first_point_normalize(inplace=True)

        self.latitude -= self.latitude[0]
        self.longitude -= self.longitude[0]
        self.time -= self.time[0]

        return self
    def normalize(self, type='FirstPoint', inplace=True):
        if type == 'FirstPoint':
            return self._first_point_normalize(inplace=inplace)
        else:
            raise ValueError(f"Unknown normalization '{type}'")

