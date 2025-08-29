from ray import *

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
    max_distance: float
) -> Image:
    right = normalize(cross(direction, up))
    up = normalize(cross(right, direction))
    image = Image.new("RGB", (width, height))
    for y in range(-height // 2, height // 2):
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
    return image


if __name__ == "__main__":
    origin = (0, 3, 4)
    direction = normalize((0, -3, -4))
    up = (0, 0, 1)
    width = 800
    height = 600
    sdf = smooth_union(
        torus(1, 0.5),
        shifted(
            rotated(torus(1, 0.5), (1, 0, 0), pi / 2),
            (1, 0, 0),
        ),
        0.5,
    )
    image = render(
        sdf,
        origin=origin,
        direction=direction,
        up=up,
        width=width,
        height=height,
        focal_length=500.0,
        eps=1e-3,
        max_distance=100.0,
    )
    image.show()
