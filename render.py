from math import cos, sin, pi, dist

from ray import Ray
from geo import Point, SurfacePoint, project_to_surface, connect
from sdf import SDF, torus, shifted, rotated, smooth_union
from point import normalize, cross, add, mul, vec, dot

from PIL import Image


def render(
    sdf: SDF,
    *,
    origin: Point,
    direction: Point,
    up: Point,
    width: int,
    height: int,
    focal_length: float,
    eps: float,
    max_distance: float,
    points: list[SurfacePoint],
) -> Image:
    right = normalize(cross(direction, up))
    up = normalize(cross(right, direction))
    image = Image.new("RGB", (width, height))
    for y in range(-height // 2, height // 2):
        if y % 20 == 0:
            print(y)
        for x in range(-width // 2, width // 2):
            ray_direction = normalize(
                add(
                    direction,
                    add(mul(right, x / focal_length), mul(up, y / focal_length)),
                )
            )
            ray = Ray(origin, ray_direction)
            hit = ray.propagate(sdf, eps=eps, max_distance=max_distance)
            if hit.hit:
                image.putpixel(
                    (x + width // 2, y + height // 2),
                    (
                        int((hit.normal[0] + 1) * 128),
                        int((hit.normal[1] + 1) * 128),
                        int((hit.normal[2] + 1) * 128),
                    ),
                )
            else:
                image.putpixel((x + width // 2, y + height // 2), (0, 0, 0))
    for pt in points:
        v = normalize(vec(origin, pt.point))
        if dot(v, pt.normal) > 0:
            continue
        ray = Ray(origin, v)
        hit = ray.propagate(sdf, eps=eps, max_distance=max_distance)
        if not hit.hit:
            continue
        if dist(ray.origin, pt.point) > 4 * eps:
            continue
        a = 1 / dot(v, direction)
        x = int(dot(v, right) * focal_length * a)
        y = int(dot(v, up) * focal_length * a)
        image.putpixel((x + width // 2, y + height // 2), (255, 0, 0))
    return image


if __name__ == "__main__":
    origin = (0, 3, 4)
    direction = normalize((0, -3, -4))
    up = (0, 0, 1)
    width = 800
    height = 600
    surface_sdf = smooth_union(
        shifted(
            torus(1, 0.5),
            (-0.5, 0, 0),
        ),
        shifted(
            rotated(torus(1, 0.5), (1, 0, 0), pi / 2),
            (0.5, 0, 0),
        ),
        0.5,
    )
    path = []
    for b in [
        # (-2, -2, -2),
        # (2, -2, -2),
        # (-2, 2, -2),
        # (2, 2, -2),
        # (-2, -2, 2),
        # (2, -2, 2),
        (-2, 2, 2),
        (2, 2, 2),
    ]:
        for a in [pi / 2 * i for i in range(4)]:
            surf_pt = project_to_surface(
                surface_sdf,
                p=b,
                direction=(cos(a), sin(a), 0),
            )
            for _ in range(100):
                surf_pt = surf_pt.move(0.005)
                path.append(surf_pt)
    print("Connecting...")
    path += connect(surface_sdf, (-2, 2, 2), (2, 2, 2))
    image = render(
        surface_sdf,
        origin=origin,
        direction=direction,
        up=up,
        width=width,
        height=height,
        focal_length=500.0,
        eps=1e-3,
        max_distance=100.0,
        points=path,
    )
    image.show()
