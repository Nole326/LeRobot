"""Materialize any catalogue ID as a portable USD layer over its measured geometry."""

import argparse
import math
import os
from itertools import product
from pathlib import Path

from .build_object_scenes import COLORS as RGB
from .build_object_scenes import build_scene, cached_asset

COLORS = (
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "black",
    "brown",
    "cyan",
    "violet",
    "gray",
    "pink",
    "ivory",
)
LOOKS = (
    "solid",
    "stripes",
    "cross_stripes",
    "checker",
    "gradient",
    "diagonal",
    "dots",
    "waves",
    "dark",
    "matte",
    "gloss",
)
MASSES = (8, 9, 10, 11, 12)
SEEDS = (42000, 42001)
MODES = ("front_rgb", "privileged_state")


def profiles():
    groups = {}
    for shape, base in [
        ("cube", (30, 30, 30)),
        ("sphere", (30, 30, 30)),
        ("box", (36, 22, 30)),
        ("cylinder", (24, 24, 30)),
        ("ellipsoid", (36, 22, 30)),
        ("square_pyramid", (30, 30, 30)),
        ("irregular_prism", (33, 25, 30)),
        ("triangular_prism", (34, 26, 30)),
    ]:
        groups[shape] = [
            tuple(round(a * s, 5) for a in base) for s in (0.85, 0.90, 1.0, 1.10, 1.15)
        ]
    groups["box"] += list(product((32, 36, 40), (18, 22, 28), (27, 30, 33)))
    groups["cylinder"] += [(d, d, h) for d, h in product((20, 24, 28), (24, 30, 36))]
    groups["ellipsoid"] += list(product((32, 36, 45), (20, 22, 26), (25, 30, 34)))
    groups["square_pyramid"] += [
        (d, d, h) for d, h in product((24, 30, 34), (24, 30, 36))
    ]
    for shape in ("irregular_prism", "triangular_prism"):
        x, y, z = groups[shape][2]
        groups[shape] += [
            (x * s, y * s, h) for s, h in product((0.85, 1.0, 1.15), (24, 30, 36))
        ]
    for n in range(3, 9):
        angles = [math.tau * i / n for i in range(n)]
        cx, cy = zip(*[(math.cos(a), math.sin(a)) for a in angles])
        groups[f"prism{n}"] = [
            (d * (max(cx) - min(cx)) / 2, d * (max(cy) - min(cy)) / 2, h)
            for d, h in product((20, 24, 28), (24, 30, 36))
        ]
    # Historical hexagon and octagon orientations and nominal AABBs are explicit controls.
    for shape in ("hexagonal_prism", "octagonal_prism"):
        groups[shape] = [(d, d, h) for d, h in product((20, 24, 28), (24, 30, 36))]
    return [
        (s, d)
        for s, ds in groups.items()
        for d in dict.fromkeys(tuple(round(x, 6) for x in v) for v in ds)
    ]


PROFILES = profiles()
SIZE = len(PROFILES) * len(COLORS) * len(LOOKS) * len(MASSES)


def spec(index):
    if not 0 <= index < SIZE:
        raise ValueError("Object index out of range")
    index, g = divmod(index, len(PROFILES))
    index, c = divmod(index, len(COLORS))
    m, p = divmod(index, len(LOOKS))
    shape, dims = PROFILES[g]
    return g, shape, dims, COLORS[c], LOOKS[p], MASSES[m]


def materialize(index, target, assets):
    g, shape, dims, color, look, mass = spec(index)
    target = Path(target).resolve()
    assets = Path(assets).resolve()
    rel = lambda p: Path(os.path.relpath(p, target.parent)).as_posix()
    base = rel(assets / "templates" / f"g{g:03}.usda")
    texture = rel(assets / "textures" / f"{color}-{look}.png")
    with target.open("x", encoding="utf8") as scene:
        scene.write(layer_text(shape, dims, color, look, mass, base, texture))


def layer_text(shape, dims, color, look, mass, base, texture):
    roughness = {"matte": 0.9, "gloss": 0.12}.get(look, 0.5)
    return f'''#usda 1.0
(
    defaultPrim = "world"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [@{base}@]
    customLayerData = {{
        double reset_z_offset_m = {(dims[2] - 30) / 2000:.12g}
        double mass_g = {mass}
        string liftcube_shape = "{shape}"
        string liftcube_color = "{color}"
        string pattern = "{look}"
    }}
)
over "world" {{
    over "cube" {{
        over "cube" {{
            float physics:mass = {mass / 1000}
            over "TextureMaterial" {{
                over "Surface" {{
                    float inputs:roughness = {roughness}
                }}
                over "Texture" {{
                    asset inputs:file = @{texture}@
                }}
            }}
        }}
    }}
}}
'''


def generate(index, source, target):
    """Generate the selected scene and its adjacent shareable dependencies."""
    from .object_textures import write_texture

    g, shape, dims, color, look, mass = spec(index)
    source = Path(source).resolve(strict=True)
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    # A change of course source must never silently reuse another scene's template.
    import hashlib

    source_key = hashlib.sha256(os.fsencode(source) + source.read_bytes()).hexdigest()[
        :16
    ]
    assets = target.parent / ("atlas-" + source_key)
    template = assets / "templates" / f"g{g:03}.usda"
    with cached_asset(template) as pending:
        if pending is not None:
            build_scene(
                source, pending, shape, "red", pattern="diagonal", dimensions_mm=dims
            )
    write_texture(assets / "textures", look, color, RGB[color])
    materialize(index, target, assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Course table_with_cube/scene.usd"
    )
    parser.add_argument(
        "--id", type=int, required=True, choices=range(SIZE), metavar="ID"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.id, args.source, args.output)


if __name__ == "__main__":
    main()
