from math import cos, sin, pi, dist

from ray import Ray
from geo import Point, SurfacePoint, project_to_surface, connect
from sdf import SDF, sphere, torus, shifted, rotated, smooth_union
from point import normalize, cross, add, mul, vec, dot
from cloud import create_cloud
from triangulate import triangulate

from PIL import Image, ImageDraw


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
    marks: list[Point],
    triangles: list[tuple[int, int, int]],
    render_surface: bool = True,
) -> Image:
    right = normalize(cross(direction, up))
    up = normalize(cross(right, direction))
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    if render_surface:
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

    def on_screen(p: SurfacePoint) -> tuple[int, int, bool]:
        v = normalize(vec(origin, p.point))
        a = 1 / dot(v, direction)
        x = int(dot(v, right) * focal_length * a)
        y = int(dot(v, up) * focal_length * a)
        return x, y, dot(v, p.normal) < 0

    triangle_vertices = set()
    for tri in triangles:
        for v in tri:
            triangle_vertices.add(v)
        for p, q in [(marks[tri[0]], marks[tri[1]]), (marks[tri[1]], marks[tri[2]]), (marks[tri[2]], marks[tri[0]])]:
            x0, y0, behind0 = on_screen(p)
            x1, y1, behind1 = on_screen(q)
            if not behind0 and not behind1:
                draw.line((x0 + width // 2, y0 + height // 2, x1 + width // 2, y1 + height // 2), fill=(255, 255, 255))

    for i, mark in enumerate(marks):
        x, y, behind = on_screen(mark)
        if behind:
            continue
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                image.putpixel(
                    (x + width // 2 + dx, y + height // 2 + dy),
                    (255 * (i in triangle_vertices), 255 * behind, 255 * (not behind)),
                )

    for pt in points:
        x, y, behind = on_screen(pt)
        if behind:
            continue
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
            (-1, 0, 0),
        ),
        shifted(
            rotated(torus(1, 0.5), (1, 0, 0), pi / 2),
            (1, 0, 0),
        ),
        0.5,
    )
    # surface_sdf = sphere((0, 0, 0), 1)
    path = []
    marks = create_cloud(
        surface_sdf, num_points=300, near_dist=0.7, step_size=0.02, num_steps=100
    )
    dists = []
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            dists.append(dist(marks[i].point, marks[j].point))
    print(min(dists), max(dists))
    triangles = triangulate(marks, near_dist=0.7)
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
        marks=marks,
        triangles=triangles,
        render_surface=True,
    )
    image.show()
