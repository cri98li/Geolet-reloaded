from typing import Any

import numpy as np
from geolib import geohash
from tqdm.auto import tqdm

from geoletrld.partitioners.PartitionerInterface import PartitionerInterface
from geoletrld.utils.Trajectories import Trajectories
from geoletrld.utils.Trajectory import Trajectory


class GeohashPartitioner(PartitionerInterface):
    def __init__(self, precision, verbose=True):
        self.precision = precision
        self.verbose = verbose

    def transform(self, X: Trajectories[Any, Trajectory]) -> Trajectories[Any, Trajectory]:
        candidate_geolet = Trajectories[Any, Trajectory]()

        if self.verbose:
            print(F"Encoding {len(X)} trajectories with precision {self.precision}", flush=True)

        for k, v in tqdm(X.items(), disable=not self.verbose):
            latitude = v.latitude
            longitude = v.longitude
            time = v.time

            prev_encode = geohash.encode(latitude[0], longitude[0], self.precision)
            curr_trj = [(latitude[0], longitude[0], time[0])]
            count = 0
            for lat, lon, t in zip(latitude[1:], longitude[1:], time[1:]):
                encode = geohash.encode(lat, lon, self.precision)
                if prev_encode != encode:
                    candidate_geolet[f"{count}_{k}"] = Trajectory(values=np.array(curr_trj).T)
                    curr_trj = []
                    count += 1

                curr_trj += [(lat, lon, t)]
                prev_encode = encode

            candidate_geolet[f"{count}_{k}"] = Trajectory(values=np.array(curr_trj).T)

        return candidate_geolet.remove_singleton(inplace=True)


