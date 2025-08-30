from dataclasses import dataclass
from math import dist

from point import Point, vec, dot, cross, normalize
from geo import SurfacePoint


@dataclass
class Edge:
    a_idx: int
    b_idx: int
    idx_on_other_side: int


def is_point_on_other_side(
    p: Point, edge_a: Point, edge_b: Point, one_side: Point
) -> bool:
    one_normal = cross(vec(edge_a, edge_b), vec(edge_a, one_side))
    this_normal = cross(vec(edge_a, edge_b), vec(edge_a, p))
    return dot(one_normal, this_normal) < 0


def triangulate(
    pts: list[SurfacePoint], near_dist: float
) -> list[tuple[int, int, int]]:
    rightmost_point_index = max(range(len(pts)), key=lambda i: pts[i].point[0])
    next_rightmost_point_index = max(
        (i for i in range(len(pts)) if i != rightmost_point_index),
        key=lambda i: pts[i].point[0],
        default=None,
    )
    third_rightmost_point_index = max(
        (
            i
            for i in range(len(pts))
            if i != rightmost_point_index and i != next_rightmost_point_index
        ),
        key=lambda i: pts[i].point[0],
        default=None,
    )

    triangles = []
    edge_to_other_side: dict[tuple[int, int], int] = {}
    edges_to_do = []

    def make_edge(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    def add_edge(a: int, b: int, other_side: int) -> None:
        edge = make_edge(a, b)
        if edge in edges_to_do:
            edges_to_do.remove(edge)
            return
        if edge in edge_to_other_side:
            return
        edge_to_other_side[edge] = other_side
        edges_to_do.append(edge)

    def add_triangle(a: int, b: int, c: int) -> None:
        add_edge(a, b, c)
        add_edge(b, c, a)
        add_edge(c, a, b)
        side_a = vec(pts[b].point, pts[c].point)
        side_b = vec(pts[a].point, pts[c].point)
        if dot(pts[c].normal, cross(side_a, side_b)) < 0:
            triangles.append((a, b, c))
        else:
            triangles.append((b, a, c))

    def triangle_exists(a: int, b: int, c: int) -> bool:
        return (
            edge_to_other_side.get(make_edge(a, b)) == c
            or edge_to_other_side.get(make_edge(b, c)) == a
            or edge_to_other_side.get(make_edge(c, a)) == b
        )

    add_triangle(
        rightmost_point_index, next_rightmost_point_index, third_rightmost_point_index
    )
    while edges_to_do:
        edge = edges_to_do.pop(0)
        print(f"Num triangles: {len(triangles)}, edges remaining: {len(edges_to_do)}")
        a_idx, b_idx = edge
        other_side_idx = edge_to_other_side.get(edge)
        a = pts[a_idx].point
        b = pts[b_idx].point
        c = pts[other_side_idx].point

        smallest_dot_product = None
        best_idx = None

        pts_considered = 0
        pts_on_other_side = 0
        triangle_points_skipped = 0
        for i, d_sfpt in enumerate(pts):
            d = d_sfpt.point
            if i == a_idx or i == b_idx or i == other_side_idx:
                continue
            if not (dist(d, a) < near_dist or dist(d, b) < near_dist):
                continue
            if triangle_exists(a_idx, b_idx, i):
                triangle_points_skipped += 1
                continue
            pts_considered += 1
            if is_point_on_other_side(d, a, b, c):
                pts_on_other_side += 1
                cos_angle = dot(normalize(vec(d, a)), normalize(vec(d, b)))
                if smallest_dot_product is None or cos_angle < smallest_dot_product:
                    smallest_dot_product = cos_angle
                    best_idx = i
        print(
            f"Considered {pts_considered} points, {pts_on_other_side} on other side, {triangle_points_skipped=}"
        )
        if best_idx is not None:
            add_triangle(a_idx, b_idx, best_idx)
        else:
            raise RuntimeError("No next point found for edge")

    return triangles
