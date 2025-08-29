from math import dist
from typing import Tuple, Callable

Point = Tuple[float, float, float]
SDF = Callable[[Point], float]


def vec(p1: Point, p2: Point) -> Point:
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])


def sphere(center: Point, r: float) -> SDF:
    def sdf(p: Point) -> float:
        return dist(p, center) - r

    return sdf


def torus(center: Point, r1: float, r2: float) -> SDF:
    def sdf(p: Point) -> float:
        d = dist(p, center) - r1
        return dist((d, 0, 0), (0, 0, r2)) - r2

    return sdf


def cube(center: Point, size: float) -> SDF:
    def sdf(p: Point) -> float:
        d = vec(p, center)
        return max(abs(d[0]), max(abs(d[1]), abs(d[2]))) - size

    return sdf


def union(sdf1: SDF, sdf2: SDF) -> SDF:
    def sdf(p: Point) -> float:
        return min(sdf1(p), sdf2(p))

    return sdf


def intersection(sdf1: SDF, sdf2: SDF) -> SDF:
    def sdf(p: Point) -> float:
        return max(sdf1(p), sdf2(p))

    return sdf


def smooth_union(sdf1: SDF, sdf2: SDF, k: float) -> SDF:
    def sdf(p: Point) -> float:
        d1 = sdf1(p)
        d2 = sdf2(p)
        return min(d1, d2) - k * (d1 + d2)

    return sdf


def smooth_intersection(sdf1: SDF, sdf2: SDF, k: float) -> SDF:
    def sdf(p: Point) -> float:
        d1 = sdf1(p)
        d2 = sdf2(p)
        return max(d1, d2) + k * (d1 + d2)

    return sdf
