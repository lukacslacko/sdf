from typing import Callable
from math import hypot, dist

from point import Point, add, vec, normalize, rotate

SDF = Callable[[Point], float]


def normal(p: Point, sdf: SDF, eps: float = 1e-6) -> Point:
    dx = sdf(add(p, (eps, 0, 0))) - sdf(add(p, (-eps, 0, 0)))
    dy = sdf(add(p, (0, eps, 0))) - sdf(add(p, (0, -eps, 0)))
    dz = sdf(add(p, (0, 0, eps))) - sdf(add(p, (0, 0, -eps)))
    return normalize((dx, dy, dz))


def sphere(center: Point, r: float) -> SDF:
    def sdf(p: Point) -> float:
        return dist(p, center) - r

    return sdf


def torus(r1: float, r2: float) -> SDF:
    def sdf(p: Point) -> float:
        return hypot(hypot(p[0], p[2]) - r1, p[1]) - r2

    return sdf


def shifted(sdf: SDF, offset: Point) -> SDF:
    def sdf_shifted(p: Point) -> float:
        return sdf(add(p, offset))

    return sdf_shifted


def rotated(sdf: SDF, axis: Point, angle: float) -> SDF:
    def sdf_rotated(p: Point) -> float:
        return sdf(rotate(p, axis, angle))

    return sdf_rotated


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


def mix(a: float, b: float, t: float) -> float:
    return a * (1 - t) + b * t


def clamp(x: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(x, max_val))


def smooth_union(sdf1: SDF, sdf2: SDF, k: float) -> SDF:
    def sdf(p: Point) -> float:
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return mix(d2, d1, h) - k * h * (1.0 - h)

    return sdf


def smooth_intersection(sdf1: SDF, sdf2: SDF, k: float) -> SDF:
    def sdf(p: Point) -> float:
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return mix(d2, d1, h) + k * h * (1.0 - h)

    return sdf


def smooth_subtraction(sdf1: SDF, sdf2: SDF, k: float) -> SDF:
    def sdf(p: Point) -> float:
        d1 = sdf1(p)
        d2 = sdf2(p)
        h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
        return mix(d2, -d1, h) + k * h * (1.0 - h)

    return sdf
