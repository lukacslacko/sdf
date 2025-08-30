from random import uniform
from math import dist, hypot

from sdf import SDF
from point import Point, normalize, mul, vec, add, add_mul
from geo import project_to_surface, SurfacePoint


def create_cloud(
    sdf: SDF, num_points: int, near_dist: float, num_steps: int, step_size: float
) -> list[SurfacePoint]:
    """Create a point cloud on the surface given by sdf.

    Initialize with randomly projected points, then move each point away from other
    points within `near_dist` and reproject them onto the surface. For non-convex
    shapes `near_dist` should be on the order of the size of local convexity.
    """

    def randpoint() -> Point:
        return mul(normalize((uniform(-2, 2), uniform(-2, 2), uniform(-2, 2))), 5)

    points = [
        project_to_surface(sdf, p=randpoint(), direction=(1, 0, 0))
        for _ in range(num_points)
    ]
    for step in range(num_steps):
        total_movement = 0
        for i in range(num_points):
            pt = points[i]
            move_vec = (0, 0, 0)
            for j in range(num_points):
                if i == j:
                    continue
                v = vec(points[j].point, pt.point)
                if hypot(*v) < near_dist:
                    move_vec = add(move_vec, mul(v, 1 / hypot(*v) ** 2))
            if move_vec == (0, 0, 0):
                continue
            guess = add_mul(
                pt.point, move_vec, step_size * (num_steps - step) / num_steps
            )
            new_point = project_to_surface(sdf, p=guess, direction=(1, 0, 0))
            total_movement += dist(pt.point, new_point.point)
            points[i] = new_point
        print(f"Step {step + 1}/{num_steps} total movement: {total_movement}")
    return points
