from math import cos, sin, pi, dist

from ray import Ray
from geo import Point, SurfacePoint, project_to_surface, connect
from sdf import SDF, sphere, torus, shifted, rotated, smooth_union
from point import normalize, cross, add, mul, vec, dot
from cloud import create_cloud
from triangulate import Triangle, triangulate
from stippling import make_surface, stipple

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
    triangles: list[Triangle],
    render_surface: bool = True,
    show_surface: bool = True,
) -> Image:
    right = normalize(cross(direction, up))
    up = normalize(cross(right, direction))
    if render_surface or not show_surface:
        image = Image.new("RGB", (width, height))
    else:
        image = Image.open("background.png")
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
        ray = Ray(origin, v)
        hit = ray.propagate(sdf, eps=eps, max_distance=max_distance)
        a = 1 / dot(v, direction)
        x = int(dot(v, right) * focal_length * a)
        y = int(dot(v, up) * focal_length * a)
        return x, y, dist(p.point, hit.point) > 10 * eps if hit.hit else True

    triangle_vertices = set()
    for tri in triangles:
        triangle_vertices.add(tri.a_idx)
        triangle_vertices.add(tri.b_idx)
        triangle_vertices.add(tri.c_idx)
        for p, q in [
            (marks[tri.a_idx], marks[tri.b_idx]),
            (marks[tri.b_idx], marks[tri.c_idx]),
            (marks[tri.c_idx], marks[tri.a_idx]),
        ]:
            x0, y0, behind0 = on_screen(p)
            x1, y1, behind1 = on_screen(q)
            if not behind0 and not behind1:
                draw.line(
                    (
                        x0 + width // 2,
                        y0 + height // 2,
                        x1 + width // 2,
                        y1 + height // 2,
                    ),
                    fill=(255, 255, 255),
                )

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

    render_params = {
        "sdf": surface_sdf,
        "origin": origin,
        "direction": direction,
        "up": up,
        "width": width,
        "height": height,
        "focal_length": 500.0,
        "eps": 1e-3,
        "max_distance": 100.0,
    }

    rerender_background = False
    if rerender_background:
        background = render(
            **render_params,
            points=[],
            marks=[],
            triangles=[],
            render_surface=True,
        )
        background.save("background.png")
    else:
        background = Image.open("background.png")

    path = []
    cloud = create_cloud(
        surface_sdf, num_points=300, near_dist=0.7, step_size=0.02, num_steps=100
    )
    dists = []
    for i in range(len(cloud)):
        for j in range(i + 1, len(cloud)):
            dists.append(dist(cloud[i].point, cloud[j].point))
    print(min(dists), max(dists))
    triangles = triangulate(cloud, near_dist=0.7)
    num_edges = len(
        {
            (min(a, b), max(a, b))
            for tri in triangles
            for a, b in [
                (tri.a_idx, tri.b_idx),
                (tri.b_idx, tri.c_idx),
                (tri.c_idx, tri.a_idx),
            ]
        }
    )
    num_vertices = len(cloud)
    num_faces = len(triangles)
    print(f"Vertices: {num_vertices}, Edges: {num_edges}, Faces: {num_faces}")
    print(f"Euler characteristic: {num_vertices - num_edges + num_faces}")
    image = render(
        **render_params,
        points=path,
        marks=cloud,
        triangles=triangles,
        render_surface=False,
        show_surface=True,
    )
    image.save("mesh.png")

    surface = make_surface(surface_sdf, cloud, triangles)
    stippled_points = stipple(surface, num_points=100, num_iters=100000)

    for i in range(len(stippled_points)):
        print(f"Connecting point {i + 1}/{len(stippled_points)}")
        best_path = []
        nearest_dist = 1.5
        for j in range(len(stippled_points)):
            print(j)
            if i == j:
                continue
            if dist(stippled_points[i].point, stippled_points[j].point) > nearest_dist:
                continue
            pts, dst = connect(
                surface_sdf, stippled_points[i].point, stippled_points[j].point
            )
            if dst < nearest_dist:
                nearest_dist = dst
                best_path = pts
        path += best_path

    image = render(
        **render_params,
        points=path,
        marks=stippled_points,
        triangles=[],
        render_surface=False,
        show_surface=True,
    )
    image.save("stippled.png")
