from math import dist
from typing import Tuple

Point = Tuple[float, float, float]


def vec(p1: Point, p2: Point) -> Point:
    return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])


def cross(p1: Point, p2: Point) -> Point:
    return (
        p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p1[1] * p2[0],
    )


def length(p: Point) -> float:
    return dist(p, (0, 0, 0))


def mul(p: Point, scalar: float) -> Point:
    return (p[0] * scalar, p[1] * scalar, p[2] * scalar)


def add(p1: Point, p2: Point) -> Point:
    return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])


def add_mul(p1: Point, p2: Point, scalar: float) -> Point:
    return (p1[0] + p2[0] * scalar, p1[1] + p2[1] * scalar, p1[2] + p2[2] * scalar)


def normalize(p: Point) -> Point:
    return (p[0] / length(p), p[1] / length(p), p[2] / length(p))
