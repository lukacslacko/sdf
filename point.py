from math import dist, cos, sin, pi, hypot
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


def dot(v1: Point, v2: Point) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def orthogonal(v: Point, n: Point) -> Point:
    n = normalize(n)
    d = dot(v, n)
    return normalize((v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]))


def rotate(p: Point, axis: Point, angle: float) -> Point:
    ux, uy, uz = axis
    x, y, z = p
    cos_a = cos(angle)
    sin_a = sin(angle)
    return (
        (cos_a + (1 - cos_a) * ux * ux) * x
        + ((1 - cos_a) * ux * uy - uz * sin_a) * y
        + ((1 - cos_a) * ux * uz + uy * sin_a) * z,
        ((1 - cos_a) * uy * ux + uz * sin_a) * x
        + (cos_a + (1 - cos_a) * uy * uy) * y
        + ((1 - cos_a) * uy * uz - ux * sin_a) * z,
        ((1 - cos_a) * uz * ux - uy * sin_a) * x
        + ((1 - cos_a) * uz * uy + ux * sin_a) * y
        + (cos_a + (1 - cos_a) * uz * uz) * z,
    )


def mul(p: Point, scalar: float) -> Point:
    return (p[0] * scalar, p[1] * scalar, p[2] * scalar)


def add(p1: Point, p2: Point) -> Point:
    return (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])


def add_mul(p1: Point, p2: Point, scalar: float) -> Point:
    return (p1[0] + p2[0] * scalar, p1[1] + p2[1] * scalar, p1[2] + p2[2] * scalar)


def normalize(p: Point) -> Point:
    return (p[0] / hypot(*p), p[1] / hypot(*p), p[2] / hypot(*p))
