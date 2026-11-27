"""Visual-only UV patterns that follow the existing LiftCube collision surface."""

import math
from pathlib import Path

import numpy as np

from .build_object_scenes import cached_asset


def write_texture(directory, pattern, color, rgb):
    from PIL import Image

    path = directory / f"{color}-{pattern}.png"
    with cached_asset(path) as pending:
        if pending is not None:
            Image.fromarray(texture_pixels(pattern, rgb)).save(pending)
    return path


def texture_pixels(pattern, rgb):
    u, v = np.meshgrid((np.arange(256) + 0.5) / 256, (np.arange(256) + 0.5) / 256)
    accent = {
        "solid": 0,
        "dark": 0,
        "matte": 0,
        "gloss": 0,
        "stripes": (4 * u) % 1 < 0.45,
        "cross_stripes": (4 * v) % 1 < 0.45,
        "checker": (np.floor(6 * u) + np.floor(6 * v)) % 2,
        "gradient": np.floor(6 * u) / 5,
        "diagonal": ((u + v) * 2) % 1 < 0.32,
        "dots": ((2 * u) % 1 - 0.5) ** 2 + ((2 * v) % 1 - 0.5) ** 2 < 0.28**2,
        "waves": (2 * v + 0.24 * np.sin(2 * np.pi * u)) % 1 < 0.34,
    }[pattern]
    alpha = np.asarray(accent)[..., None]
    linear = np.broadcast_to(
        np.asarray(rgb) * (0.4 if pattern == "dark" else 1) * (1 - alpha)
        + 0.85 * alpha,
        (*u.shape, 3),
    )
    srgb = np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    return np.uint8(np.rint(srgb * 255))


def textured_surface(stage, body, visual, shape, output, pattern, color, rgb):
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from starter.envs.contact_surfaces import _mesh_cylinder, _mesh_sphere

    from .build_object_scenes import render_material

    transform = np.asarray(
        UsdGeom.XformCache().ComputeRelativeTransform(visual, body)[0]
    )
    if shape == "box":
        half = UsdGeom.Cube(visual).GetSizeAttr().Get() / 2
        points = half * np.array(
            [
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, -1, -1),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 1),
            ]
        )
        faces = [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [1, 5, 7, 3],
            [0, 2, 6, 4],
        ]
    else:
        source = visual
        if shape in ("sphere", "cylinder"):
            # Use the same triangulation as the live contact layer.
            temporary = Usd.Stage.CreateInMemory()
            if shape == "sphere":
                prim = UsdGeom.Sphere.Define(temporary, "/Object")
                prim.CreateRadiusAttr(UsdGeom.Sphere(visual).GetRadiusAttr().Get())
                _mesh_sphere(temporary, prim.GetPrim())
            else:
                prim = UsdGeom.Cylinder.Define(temporary, "/Object")
                prim.CreateRadiusAttr(UsdGeom.Cylinder(visual).GetRadiusAttr().Get())
                prim.CreateHeightAttr(UsdGeom.Cylinder(visual).GetHeightAttr().Get())
                prim.CreateAxisAttr("Z")
                _mesh_cylinder(temporary, prim.GetPrim())
            source = temporary.GetPrimAtPath("/Object")
        mesh = UsdGeom.Mesh(source)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        faces = np.split(
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), counts.cumsum()[:-1]
        )
    points = points @ transform[:3, :3] + transform[3, :3]
    uvw = (points - points.min(axis=0)) / np.ptp(points, axis=0)
    uvs = []
    for face in faces:
        v = points[face]
        normal = np.cross(v[1] - v[0], v[2] - v[0])
        axis = int(np.argmax(abs(normal)))
        if shape == "sphere" or (shape == "cylinder" and axis != 2):
            u = np.arctan2(v[:, 1], v[:, 0]) / (2 * math.pi) + 0.5
            if np.ptp(u) > 0.5:
                u[u < 0.5] += 1
            uv = np.column_stack((2 * u, uvw[face, 2]))
        else:
            uv = uvw[face][:, [(axis + 1) % 3, (axis + 2) % 3]]
        uvs.extend(uv.tolist())
    skin = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("PatternSurface"))
    skin.CreatePointsAttr(points.tolist())
    skin.CreateFaceVertexCountsAttr([len(f) for f in faces])
    skin.CreateFaceVertexIndicesAttr(np.concatenate(faces).tolist())
    skin.CreateSubdivisionSchemeAttr("none")
    if shape == "sphere":
        skin.CreateNormalsAttr(
            (points / np.linalg.norm(points, axis=1, keepdims=True)).tolist()
        )
        skin.SetNormalsInterpolation("vertex")
    st = UsdGeom.PrimvarsAPI(skin).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    )
    st.Set(uvs)

    file = write_texture(Path(output).parent / "texture-patterns", pattern, color, rgb)
    path = body.GetPath().AppendChild("TextureMaterial")
    mat = render_material(stage, path, rgb)
    reader = UsdShade.Shader.Define(stage, path.AppendChild("UV"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, path.AppendChild("Texture"))
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("texture-patterns/" + file.name)
    )
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    for key in ("wrapS", "wrapT"):
        tex.CreateInput(key, Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader = UsdShade.Shader.Get(stage, path.AppendChild("Surface"))
    shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    UsdShade.MaterialBindingAPI.Apply(skin.GetPrim()).Bind(mat)
    UsdGeom.Imageable(visual).MakeInvisible()
