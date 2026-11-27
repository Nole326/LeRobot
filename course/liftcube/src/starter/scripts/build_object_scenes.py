"""Build optional USD layers over the course LiftCube scene. Run with Isaac Python or usd-core."""

import math
import os
from pathlib import Path

COLORS = {
    "red": (0.8, 0.025, 0.025),
    "green": (0.03, 0.65, 0.06),
    "blue": (0.025, 0.10, 0.8),
    "yellow": (0.85, 0.65, 0.025),
    "orange": (0.95, 0.21, 0.02),
    "cyan": (0.02, 0.65, 0.73),
    "violet": (0.45, 0.05, 0.72),
    "pink": (0.95, 0.22, 0.45),
    "ivory": (0.85, 0.80, 0.63),
}
COLORS.update({"gray": (0.28, 0.28, 0.28), "brown": (0.32, 0.14, 0.055)})
DIMENSIONS_MM = {
    "cube": (30, 30, 30),
    "box": (36, 22, 30),
    "cylinder": (24, 24, 30),
    "sphere": (30, 30, 30),
    "triangular_prism": (34, 26, 30),
    "ellipsoid": (36, 22, 30),
    "irregular_prism": (33, 25, 30),
    "square_pyramid": (30, 30, 30),
    "octagonal_prism": (24, 24, 30),
    "hexagonal_prism": (24, 24, 30),
    "slender_ellipsoid": (45, 20, 30),
}
SHAPES = tuple(DIMENSIONS_MM)
COLORS["black"] = (0.012, 0.012, 0.012)
PRISM_SIDES = {f"prism{n}": n for n in range(3, 9)}
for name, n in PRISM_SIDES.items():
    angles = [math.tau * i / n for i in range(n)]
    x, y = [math.cos(a) for a in angles], [math.sin(a) for a in angles]
    DIMENSIONS_MM[name] = (12 * (max(x) - min(x)), 12 * (max(y) - min(y)), 30)
SHAPES = tuple(DIMENSIONS_MM)
TEXTURE_PATTERNS = ("diagonal", "dots", "waves")


def convex_object(stage, path, shape, dims):
    from pxr import UsdGeom, UsdPhysics

    if shape in ("ellipsoid", "slender_ellipsoid"):
        points = [(0, 0, -1)]
        sides, rings = 16, 7
        for ring in range(1, rings + 1):
            lat = math.pi * (ring / (rings + 1) - 0.5)
            for j in range(sides):
                lon = math.tau * j / sides
                points.append(
                    (
                        math.cos(lat) * math.cos(lon),
                        math.cos(lat) * math.sin(lon),
                        math.sin(lat),
                    )
                )
        points.append((0, 0, 1))
        faces = []
        for j in range(sides):
            k = (j + 1) % sides
            faces.append((0, 1 + k, 1 + j))
            for r in range(rings - 1):
                i = 1 + r * sides
                faces.append((i + j, i + k, i + k + sides, i + j + sides))
            i = 1 + (rings - 1) * sides
            faces.append((i + j, i + k, len(points) - 1))
    elif shape == "square_pyramid":
        points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (0, 0, 1)]
        faces = [(3, 2, 1, 0), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    else:
        if shape in PRISM_SIDES:
            n = PRISM_SIDES[shape]
            vertices = [
                (math.cos(math.tau * i / n), math.sin(math.tau * i / n))
                for i in range(n)
            ]
            spans = [max(c) - min(c) for c in zip(*vertices)]
            profile = [(2 * x / spans[0], 2 * y / spans[1]) for x, y in vertices]
        elif shape == "triangular_prism":
            profile = [(-1, -1), (1, -1), (0, 1)]
        elif shape == "octagonal_prism":
            profile = [
                (math.cos(i * math.tau / 8), math.sin(i * math.tau / 8))
                for i in range(8)
            ]
        elif shape == "hexagonal_prism":
            angle = math.pi / 12
            profile = [
                (
                    math.cos(angle + i * math.tau / 6) / math.cos(angle),
                    math.sin(angle + i * math.tau / 6) / math.cos(angle),
                )
                for i in range(6)
            ]
        else:
            profile = [
                (-1, -0.55),
                (-0.35, -1),
                (0.85, -0.72),
                (1, 0.42),
                (0.22, 1),
                (-0.82, 0.72),
            ]
        points = [(x, y, z) for z in (-1, 1) for x, y in profile]
        n = len(profile)
        faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
        for j in range(n):
            k = (j + 1) % n
            faces.append((j, k, n + k, n + j))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [tuple(a * d / 2000 for a, d in zip(pt, dims)) for pt in points]
    )
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        "convexHull"
    )
    return mesh


def render_material(stage, path, rgb, roughness=0.5):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def build_scene(
    source,
    output,
    shape,
    color,
    *,
    mass_g=10,
    pattern="solid",
    dimensions_mm=None,
):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if shape not in SHAPES or color not in COLORS:
        raise ValueError("Unsupported shape or color")
    if mass_g not in (8, 9, 10, 11, 12) or pattern not in ("solid", *TEXTURE_PATTERNS):
        raise ValueError("Unsupported mass or template texture")
    dims = DIMENSIONS_MM[shape]
    scale = 1.0
    if dimensions_mm is not None:
        dims = list(dimensions_mm)
        if len(dims) != 3 or not all(math.isfinite(d) and d > 0 for d in dims):
            raise ValueError("Expected three positive finite dimensions in millimetres")
        if shape in ("cube", "sphere") and len(set(dims)) != 1:
            raise ValueError("Cube and sphere require equal dimensions")
        if shape == "cylinder" and dims[0] != dims[1]:
            raise ValueError("Cylinder requires equal horizontal diameters")
        if shape == "cube":
            scale = dims[0] / 30
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    original = Usd.Stage.Open(str(source))
    root = original.GetDefaultPrim()
    if not root:
        raise ValueError("The source scene has no default prim")
    stage = Usd.Stage.CreateInMemory()
    layer = stage.GetRootLayer()
    # Use an absolute path during composition, then make the exported layer relocatable.
    layer.subLayerPaths = [source.as_posix()]
    stage.SetDefaultPrim(stage.GetPrimAtPath(root.GetPath()))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    body = stage.GetPrimAtPath(root.GetPath().AppendPath("cube/cube"))
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Expected the course cube rigid body")
    visual = body.GetChild("Visuals").GetChild("cube")
    collider = body.GetChild("Collisions").GetChild("cube")
    if not visual.IsA(UsdGeom.Mesh) or not collider.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Expected the course cube visual mesh and collider")

    UsdPhysics.MassAPI.Apply(body).CreateMassAttr().Set(mass_g / 1000)
    if shape == "cube" and scale != 1:
        for surface in (visual, collider):
            UsdGeom.Xformable(surface).AddScaleOp(opSuffix="objectSize").Set(
                Gf.Vec3f(scale, scale, scale)
            )
    if shape != "cube":
        # The course scene uses metres and its 30 mm cube is centred on the rigid body.
        if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-6:
            raise ValueError("The LiftCube variants require a metre-scale scene")
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
            "physics"
        )
        path = body.GetPath().AppendChild("GraspObject")
        if shape == "sphere":
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(dims[0] / 2000)
        elif shape == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateAxisAttr("Z")
            geom.CreateHeightAttr(dims[2] / 1000)
            geom.CreateRadiusAttr(dims[0] / 2000)
        elif shape != "box":
            geom = convex_object(stage, path, shape, dims)
        else:
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
            geom.AddScaleOp().Set(Gf.Vec3f(*(d / 1000 for d in dims)))
        visual.SetActive(False)
        collider.SetActive(False)
        visual = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(visual)
        if material:
            UsdShade.MaterialBindingAPI.Apply(visual).Bind(
                material, materialPurpose="physics"
            )

    # A render material binding never replaces the separate physics-purpose binding.
    rgb = Gf.Vec3f(*(value for value in COLORS[color]))
    roughness = 0.5
    mat = render_material(
        stage, body.GetPath().AppendChild("VariantColor"), rgb, roughness
    )
    UsdShade.MaterialBindingAPI.Apply(visual).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    UsdGeom.Gprim(visual).CreateDisplayColorAttr([rgb])
    if pattern in TEXTURE_PATTERNS:
        from .object_textures import textured_surface

        textured_surface(stage, body, visual, shape, output, pattern, color, rgb)
    layer.customLayerData = {
        "liftcube_shape": shape,
        "liftcube_color": color,
        "mass_g": float(mass_g),
        "reset_z_offset_m": (dims[2] - 30) / 2000,
        "pattern": pattern,
    }
    portable = Sdf.Layer.CreateAnonymous()
    portable.TransferContent(layer)
    portable.subLayerPaths = [Path(os.path.relpath(source, output.parent)).as_posix()]
    if not portable.Export(str(output)):
        raise OSError(f"Failed to export {output}")
