from dataclasses import dataclass

from sdf import *


@dataclass
class SurfacePoint:
    point: Point
    direction: Point
    normal: Point
    sdf: SDF

    def move(self, distance: float) -> "SurfacePoint":
        return project_to_surface(
            self.sdf,
            p=add_mul(self.point, self.direction, distance),
            direction=self.direction,
        )


def project_to_surface(
    sdf: SDF, *, p: Point, direction: Point, eps: float = 1e-6
) -> SurfacePoint:
    for _ in range(100):
        d = sdf(p)
        n = normal(p, sdf, eps)
        if abs(d) < eps:
            return SurfacePoint(
                point=p, direction=orthogonal(direction, n), normal=n, sdf=sdf
            )
        p = add_mul(p, n, -d)
    raise ValueError("Could not project point to surface")
