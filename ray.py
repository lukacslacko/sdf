from .sdf import *
from .point import *

from dataclasses import dataclass


@dataclass
class Hit:
    hit: bool
    normal: Point | None = None


@dataclass
class Ray:
    origin: Point
    direction: Point

    def propagate(self, sdf: SDF, eps: float = 1e-6, max_distance: float = 1e4) -> Hit:
        if sdf(self.origin) < 0:
            return Hit(True, normal(self.origin, sdf, eps))

        distance_traveled = 0.0
        while distance_traveled < max_distance:
            step = sdf(self.origin)
            if step < eps:
                return Hit(True, normal(self.origin, sdf, eps))
            while True:
                end = add_mul(self.origin, self.direction, step)
                if sdf(end) >= 0:
                    break
                step *= 0.5
            distance_traveled += step
        return Hit(False)
