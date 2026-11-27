"""Compose LiftCube collision shapes from the rendered course geometry."""

from pathlib import Path


def _overlay(source, target):
    from pxr import Usd, UsdGeom

    original = Usd.Stage.Open(str(source))
    if not original or not original.GetDefaultPrim():
        raise ValueError(f"Missing default prim in LiftCube asset: {source}")
    overlay = Usd.Stage.CreateNew(str(target))
    layer = original.GetRootLayer()
    overlay.GetRootLayer().subLayerPaths = [layer.realPath or layer.identifier]
    overlay.SetDefaultPrim(overlay.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageMetersPerUnit(overlay, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(overlay, UsdGeom.GetStageUpAxis(original))
    return overlay


def _collision(mesh, approximation, material=None):
    from pxr import Sdf, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
    if approximation == "sdf":
        try:
            from pxr import PhysxSchema
        except ImportError:  # Standalone usd-core can author the same schema.
            mesh.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            mesh.CreateAttribute(
                "physxSDFMeshCollision:sdfResolution",
                Sdf.ValueTypeNames.Int,
                custom=False,
                variability=Sdf.VariabilityUniform,
            ).Set(256)
        else:
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh).CreateSdfResolutionAttr(
                256
            )
    if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            material, materialPurpose="physics"
        )


def _sync_object_density(body, surface):
    """Match the advertised density to the authored mass and final closed collider."""
    from math import pi

    from pxr import Gf, UsdGeom, UsdPhysics

    mass_api = UsdPhysics.MassAPI(body)
    mass = mass_api.GetMassAttr().Get()
    if not mass or mass <= 0:
        return
    if surface.IsA(UsdGeom.Cube):
        volume = UsdGeom.Cube(surface).GetSizeAttr().Get() ** 3
    elif surface.IsA(UsdGeom.Sphere):
        volume = 4 * pi * UsdGeom.Sphere(surface).GetRadiusAttr().Get() ** 3 / 3
    else:
        mesh = UsdGeom.Mesh(surface)
        points = mesh.GetPointsAttr().Get()
        if not points:
            raise ValueError(f"Empty object surface: {surface.GetPath()}")
        points = [Gf.Vec3d(p) for p in points]
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        offset, signed_volume = 0, 0.0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset : offset + count]
            a = points[face[0]]
            for i in range(1, count - 1):
                signed_volume += (
                    Gf.Dot(a, Gf.Cross(points[face[i]], points[face[i + 1]])) / 6
                )
            offset += count
        volume = abs(signed_volume)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(surface)
    volume *= abs(transform.GetDeterminant())
    if volume <= 0:
        raise ValueError(f"Object collider has zero volume: {surface.GetPath()}")
    mass_api.CreateDensityAttr(mass / volume)


def build_contact_surfaces(
    robot, scene, directory, *, solid_table=False, mesh_spheres=True
):
    """Write two USD overlays. The input layers and the robot joint drives are preserved."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    directory = Path(directory)
    robot_path, scene_path = directory / "robot.usda", directory / "scene.usda"
    arm = _overlay(robot, robot_path)
    for body in arm.GetDefaultPrim().GetChildren():
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        old, visible = body.GetChild("collisions"), body.GetChild("visuals")
        if not old or not visible:
            raise ValueError(f"Missing rigid link geometry on {body.GetPath()}")
        material, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial("physics")
        visible.SetInstanceable(False)
        meshes = [p for p in Usd.PrimRange(visible) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ValueError(f"Missing visible mesh on {body.GetPath()}")
        for mesh in meshes:
            _collision(mesh, "sdf", material)
        old.SetActive(False)
    arm.GetRootLayer().Save()

    world = _overlay(scene, scene_path)
    body = world.GetPrimAtPath("/world/cube/cube")
    table = world.GetPrimAtPath("/world/counter_right_main_group/geometry_0")
    if (
        not body
        or not body.HasAPI(UsdPhysics.RigidBodyAPI)
        or not table
        or not table.HasAPI(UsdPhysics.CollisionAPI)
    ):
        raise ValueError("Expected the course LiftCube rigid object and static table")
    # The tabletop is a closed box. Its triangulated surface is also its fixed collider.
    _collision(table, "convexHull" if solid_table else "none")
    variant = body.GetChild("GraspObject")
    if variant and variant.IsActive():
        if not variant.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError("The visible variant must also be its collision shape")
        if variant.IsA(UsdGeom.Mesh):
            _collision(variant, "sdf")
        elif variant.IsA(UsdGeom.Cylinder):
            _mesh_cylinder(world, variant)
        elif mesh_spheres and variant.IsA(UsdGeom.Sphere):
            _mesh_sphere(world, variant)
        _sync_object_density(body, world.GetPrimAtPath(variant.GetPath()))
    else:
        old = body.GetChild("Collisions")
        visual = body.GetChild("Visuals").GetChild("cube")
        original = old.GetChild("cube")
        if not visual or not visual.IsA(UsdGeom.Mesh) or not original:
            raise ValueError(
                "Missing visual mesh or collision material on the course cube"
            )
        material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(
            "physics"
        )
        _collision(visual, "sdf", material)
        old.SetActive(False)
        if "mass_g" in Usd.Stage.Open(str(scene)).GetRootLayer().customLayerData:
            _sync_object_density(body, visual)
    world.GetRootLayer().Save()
    return str(robot_path), str(scene_path)


def _mesh_cylinder(stage, prim):
    """Render and collide the same capped regular prism at the original cylinder path."""
    from math import cos, sin, tau

    from pxr import UsdGeom

    cylinder = UsdGeom.Cylinder(prim)
    radius = cylinder.GetRadiusAttr().Get()
    half_height = cylinder.GetHeightAttr().Get() / 2
    axis = ("X", "Y", "Z").index(cylinder.GetAxisAttr().Get())
    sides = 32  # At the 12 mm suite radius the distance from an ideal arc is below 0.058 mm.
    points = []
    for height in (-half_height, half_height):
        for i in range(sides):
            point = [0.0, 0.0, 0.0]
            point[axis] = height
            point[(axis + 1) % 3] = radius * cos(tau * i / sides)
            point[(axis + 2) % 3] = radius * sin(tau * i / sides)
            points.append(tuple(point))
    faces = [list(reversed(range(sides))), list(range(sides, 2 * sides))]
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, j + sides, i + sides])
    mesh = UsdGeom.Mesh.Define(stage, prim.GetPath())
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    _collision(mesh.GetPrim(), "sdf")


def _mesh_sphere(stage, prim):
    """Use common triangles for the visible surface and the signed-distance shape."""
    from math import cos, pi, sin, tau

    from pxr import UsdGeom

    radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
    slices, divisions = 48, 32
    normals = [(0.0, 0.0, -1.0)]
    for row in range(1, divisions):
        lat = pi * (row / divisions - 0.5)
        for col in range(slices):
            lon = tau * col / slices
            normals.append((cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)))
    north = len(normals)
    normals.append((0.0, 0.0, 1.0))
    faces = []

    def ring(index, col):
        return 1 + index * slices + col % slices

    for col in range(slices):
        faces.append((0, ring(0, col + 1), ring(0, col)))
        for row in range(divisions - 2):
            a, b, c, d = (
                ring(row, col),
                ring(row, col + 1),
                ring(row + 1, col + 1),
                ring(row + 1, col),
            )
            faces.extend(((a, b, c), (a, c, d)))
        faces.append((north, ring(divisions - 2, col), ring(divisions - 2, col + 1)))
    mesh = UsdGeom.Mesh.Define(stage, prim.GetPath())
    mesh.CreatePointsAttr([tuple(radius * x for x in vertex) for vertex in normals])
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    _collision(mesh.GetPrim(), "sdf")
