from math import radians, sin, asin, cos, sqrt

import numpy as np

from geoletrld.utils import Trajectory


def rotate(geolet: Trajectory, alpha: float):
    c, s = np.cos(alpha), np.sin(alpha)
    rotation_matrix = np.array(((c, -s), (s, c))).reshape(2, 2)

    rotated_lat_lon = np.dot(rotation_matrix, geolet.lat_lon)

    return Trajectory(
        latitude=rotated_lat_lon[0],
        longitude=rotated_lat_lon[1],
        time=geolet.time,
    )


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    m = 6371 * c * 1000
    return m


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_distance(a: np.ndarray, b: np.ndarray):
    return 1 - cosine_similarity(a, b)
