from dataclasses import dataclass
from math import dist

from sdf import Point, SDF, normal
from point import add_mul, orthogonal, normalize, rotate, vec

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


@dataclass
class Approach:
    error: float
    distance_traveled: float


def closest_approach(
    start: SurfacePoint, target: SurfacePoint, step_size: float
) -> Approach:
    prev = dist(start.point, target.point)
    distance_traveled = 0
    while True:
        start = start.move(step_size)
        distance_traveled += step_size
        curr = dist(start.point, target.point)
        if curr > prev:
            return Approach(prev, distance_traveled)
        prev = curr


def connect(
    sdf: SDF, p: Point, q: Point, *, step_size: float = 1e-3, eps: float = 1e-4
) -> tuple[list[SurfacePoint], float]:
    start = project_to_surface(sdf, p=p, direction=normalize(vec(p, q)), eps=eps)
    end = project_to_surface(sdf, p=q, direction=normalize(vec(q, p)), eps=eps)
    approach = closest_approach(start, end, step_size=step_size)
    ang = 0
    initial_d_ang = 0.05
    right_approach = closest_approach(
        SurfacePoint(
            point=start.point,
            normal=start.normal,
            direction=rotate(start.direction, start.normal, initial_d_ang),
            sdf=sdf,
        ),
        end,
        step_size=step_size,
    )
    left_approach = closest_approach(
        SurfacePoint(
            point=start.point,
            normal=start.normal,
            direction=rotate(start.direction, start.normal, -initial_d_ang),
            sdf=sdf,
        ),
        end,
        step_size=step_size,
    )
    d_ang = (
        -initial_d_ang if left_approach.error < right_approach.error else initial_d_ang
    )
    while abs(d_ang) > eps:
        # print(f"{ang=}, {d_ang=}, {approach=}")
        curr_ang = ang + d_ang
        curr_approach = closest_approach(
            SurfacePoint(
                point=start.point,
                normal=start.normal,
                direction=rotate(start.direction, start.normal, curr_ang),
                sdf=sdf,
            ),
            end,
            step_size=step_size,
        )
        # print(f"{curr_ang=}, {curr_approach=}")
        if curr_approach.error < approach.error:
            approach = curr_approach
            ang = curr_ang
            continue
        ang -= d_ang
        approach = closest_approach(
            SurfacePoint(
                point=start.point,
                normal=start.normal,
                direction=rotate(start.direction, start.normal, ang),
                sdf=sdf,
            ),
            end,
            step_size=step_size,
        )
        d_ang *= 0.5
    final_start = SurfacePoint(
        point=start.point,
        normal=start.normal,
        direction=rotate(start.direction, start.normal, ang),
        sdf=sdf,
    )
    path = [final_start]
    distance_traveled = 0.0
    while distance_traveled < approach.distance_traveled:
        final_start = final_start.move(step_size)
        path.append(final_start)
        distance_traveled += step_size
    return (path, distance_traveled)
