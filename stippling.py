from dataclasses import dataclass
from math import dist
from random import uniform
import bisect

from sdf import SDF
from geo import SurfacePoint, project_to_surface
from triangulate import Triangle
from point import vec, add, mul, add_mul


@dataclass
class Surface:
    sdf: SDF
    vertices: list[SurfacePoint]
    triangles: list[Triangle]
    cumulative_areas: list[float]

    def get_random_point(self) -> SurfacePoint:
        total_area = self.cumulative_areas[-1]
        r = uniform(0, total_area)
        triangle = self.triangles[bisect.bisect_right(self.cumulative_areas, r)]
        a = self.vertices[triangle.a_idx].point
        b = self.vertices[triangle.b_idx].point
        c = self.vertices[triangle.c_idx].point
        x = uniform(0, 1)
        y = uniform(0, 1)
        u = min(x, y)
        v = max(x, y)
        p = add(mul(a, u), add(mul(b, v - u), mul(c, 1 - v)))
        return project_to_surface(self.sdf, p=p, direction=(1, 0, 0))


def make_surface(
    sdf: SDF, vertices: list[SurfacePoint], triangles: list[Triangle]
) -> Surface:
    cumulative_areas = []
    total_area = 0.0
    for triangle in triangles:
        total_area += triangle.area
        cumulative_areas.append(total_area)
    return Surface(sdf, vertices, triangles, cumulative_areas)


def stipple(surface: Surface, num_points: int, num_iters: int) -> list[SurfacePoint]:
    points = [surface.get_random_point() for _ in range(num_points)]
    num_moved = [0 for _ in range(num_points)]
    for iter in range(num_iters):
        if iter % num_points == 0:
            print(
                f"Stippling iteration {iter}/{num_iters}, min moves {min(num_moved)}, max moves {max(num_moved)}"
            )
        target = surface.get_random_point()
        min_dist = float("inf")
        min_idx = None
        for i, point in enumerate(points):
            d = dist(target.point, point.point)
            if d < min_dist:
                min_dist = d
                min_idx = i
        num_moved[min_idx] += 1
        new_point = add_mul(
            points[min_idx].point,
            vec(points[min_idx].point, target.point),
            1 / (1 + num_moved[min_idx]),
        )
        points[min_idx] = project_to_surface(
            surface.sdf, p=new_point, direction=(1, 0, 0)
        )
    return points
