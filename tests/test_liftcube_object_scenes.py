"""Exercise the optional scene builder without starting Isaac."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "course/liftcube/src"))
from starter.scripts.build_object_scenes import build_scene
from starter.scripts.object_catalog import SIZE, PROFILES, spec


class ObjectCatalogueTests(unittest.TestCase):
    def test_full_grid_and_split(self):
        self.assertEqual(len(PROFILES), 196)
        self.assertEqual(SIZE, 129360)
        specs = {spec(i) for i in range(SIZE)}
        self.assertEqual(len(specs), SIZE)
        self.assertEqual(sum(s[1] in {"ellipsoid", "square_pyramid"} for s in specs), 29040)
        for index in (-1, SIZE):
            with self.assertRaises(ValueError):
                spec(index)


@unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires OpenUSD")
class ObjectSceneTests(unittest.TestCase):
    def test_scaled_box_density_preserves_mass(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from starter.envs.contact_surfaces import _sync_object_density

        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
        properties = UsdPhysics.MassAPI.Apply(body)
        properties.CreateMassAttr(.011)
        shape = UsdGeom.Cube.Define(stage, "/Body/Box")
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set((.2, .3, .4))
        shape.AddScaleOp().Set(Gf.Vec3f(.036, .022, .030))
        _sync_object_density(body, shape.GetPrim())
        self.assertAlmostEqual(properties.GetMassAttr().Get(), .011, places=8)
        self.assertAlmostEqual(properties.GetDensityAttr().Get(), .011 / (.036 * .022 * .030), places=4)

    def test_contact_overlay_uses_visible_mesh_and_preserves_source_assets(self):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        from starter.envs.contact_surfaces import build_contact_surfaces

        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            robot, world = d / "arm.usda", d / "world.usda"
            arm = Usd.Stage.CreateNew(str(robot))
            arm.SetDefaultPrim(UsdGeom.Xform.Define(arm, "/Robot").GetPrim())
            jaw = UsdGeom.Xform.Define(arm, "/Robot/jaw").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(jaw)
            UsdGeom.Mesh.Define(arm, "/Robot/jaw/visuals/finger")
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Xform.Define(arm, "/Robot/jaw/collisions").GetPrim())
            arm.GetRootLayer().Save()
            scene = Usd.Stage.CreateNew(str(world))
            scene.SetDefaultPrim(UsdGeom.Xform.Define(scene, "/world").GetPrim())
            cube = UsdGeom.Xform.Define(scene, "/world/cube/cube").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(cube)
            UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(.01)
            UsdGeom.Mesh.Define(scene, "/world/cube/cube/Visuals/cube")
            original = UsdGeom.Mesh.Define(scene, "/world/cube/cube/Collisions/cube").GetPrim()
            UsdPhysics.CollisionAPI.Apply(original)
            material = UsdShade.Material.Define(scene, "/world/friction")
            UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(original).Bind(material, materialPurpose="physics")
            table = UsdGeom.Mesh.Define(scene, "/world/counter_right_main_group/geometry_0").GetPrim()
            UsdPhysics.CollisionAPI.Apply(table)
            scene.GetRootLayer().Save()
            before = [p.read_bytes() for p in (robot, world)]
            arm_layer, scene_layer = build_contact_surfaces(robot, world, d)
            arm, scene = Usd.Stage.Open(arm_layer), Usd.Stage.Open(scene_layer)
            finger = arm.GetPrimAtPath("/Robot/jaw/visuals/finger")
            self.assertEqual(UsdPhysics.MeshCollisionAPI(finger).GetApproximationAttr().Get(), "sdf")
            self.assertFalse(arm.GetPrimAtPath("/Robot/jaw/collisions").IsActive())
            visible = scene.GetPrimAtPath("/world/cube/cube/Visuals/cube")
            self.assertTrue(visible.HasAPI(UsdPhysics.CollisionAPI))
            self.assertEqual(UsdPhysics.MeshCollisionAPI(visible).GetApproximationAttr().Get(), "sdf")
            self.assertEqual(UsdShade.MaterialBindingAPI(visible).ComputeBoundMaterial("physics")[0].GetPath(), material.GetPath())
            self.assertFalse(scene.GetPrimAtPath("/world/cube/cube/Collisions").IsActive())
            self.assertAlmostEqual(UsdPhysics.MassAPI(scene.GetPrimAtPath(cube.GetPath())).GetMassAttr().Get(), .01)
            self.assertEqual(UsdPhysics.MeshCollisionAPI(scene.GetPrimAtPath(table.GetPath())).GetApproximationAttr().Get(), "none")
            self.assertEqual(before, [p.read_bytes() for p in (robot, world)])
            solid = d / "solid"
            solid.mkdir()
            _, solid_layer = build_contact_surfaces(robot, world, solid, solid_table=True)
            solid_stage = Usd.Stage.Open(solid_layer)
            solid_mesh = solid_stage.GetPrimAtPath(table.GetPath())
            self.assertEqual(UsdPhysics.MeshCollisionAPI(solid_mesh).GetApproximationAttr().Get(), "convexHull")

    def test_cylinder_mesh_keeps_axis_dimensions_and_material_path(self):
        import math
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        from starter.envs.contact_surfaces import _mesh_cylinder

        for axis in ('X', 'Y', 'Z'):
            with self.subTest(axis=axis):
                stage = Usd.Stage.CreateInMemory()
                source = UsdGeom.Cylinder.Define(stage, '/Object')
                source.CreateAxisAttr(axis)
                source.CreateHeightAttr(.03)
                source.CreateRadiusAttr(.012)
                source.AddTranslateOp().Set((1., 2., 3.))
                mat = UsdShade.Material.Define(stage, '/Contact')
                UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(source.GetPrim()).Bind(mat, materialPurpose='physics')
                _mesh_cylinder(stage, source.GetPrim())
                mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/Object'))
                self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get(), 'sdf')
                self.assertEqual(mesh.GetLocalTransformation().ExtractTranslation(), (1., 2., 3.))
                self.assertEqual(UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial('physics')[0].GetPath(), mat.GetPath())
                coords = mesh.GetPointsAttr().Get()
                self.assertEqual(len(coords), 64)
                k = 'XYZ'.index(axis)
                for v in coords:
                    self.assertAlmostEqual(abs(v[k]), .015)
                    self.assertAlmostEqual(math.hypot(v[(k+1)%3],v[(k+2)%3]), .012)
                counts = list(mesh.GetFaceVertexCountsAttr().Get())
                self.assertEqual(sorted(counts), [4]*32+[32,32])
                self.assertEqual(mesh.GetSubdivisionSchemeAttr().Get(),'none')

    def test_sphere_mesh_shares_closed_surface_radius_and_pose(self):
        from collections import Counter
        import math
        from pxr import Usd, UsdGeom, UsdPhysics
        from starter.envs.contact_surfaces import _mesh_sphere

        stage = Usd.Stage.CreateInMemory()
        sphere = UsdGeom.Sphere.Define(stage, "/Ball")
        sphere.CreateRadiusAttr(.015)
        sphere.AddTranslateOp().Set((1., 2., 3.))
        _mesh_sphere(stage, sphere.GetPrim())
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Ball"))
        self.assertEqual(mesh.GetLocalTransformation().ExtractTranslation(), (1., 2., 3.))
        self.assertTrue(all(abs(math.hypot(*p)-.015) < 1e-8 for p in mesh.GetPointsAttr().Get()))
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        edges = Counter()
        for a,b,c in zip(faces[::3], faces[1::3], faces[2::3]):
            edges.update(((a,b), (b,c), (c,a)))
        self.assertTrue(all(count == 1 and edges[b,a] == 1 for (a,b),count in edges.items()))
        self.assertEqual(len(faces)//3, 2976)
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get(), "sdf")

    def test_recolor_and_replace_keep_the_original_body(self):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.usda"
            stage = Usd.Stage.CreateNew(str(source))
            stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/world").GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            body_path = "/world/cube/cube"
            body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.01)
            old_visual = UsdGeom.Mesh.Define(
                stage, body_path + "/Visuals/cube"
            ).GetPath()
            collider = UsdGeom.Mesh.Define(
                stage, body_path + "/Collisions/cube"
            ).GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
            material = UsdShade.Material.Define(stage, "/world/contact")
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                material, materialPurpose="physics"
            )
            stage.GetRootLayer().Save()
            original = source.read_bytes()
            for shape, type_name in (
                ("cube", "Mesh"),
                ("box", "Cube"),
                ("cylinder", "Cylinder"),
                ("sphere", "Sphere"),
            ):
                path = Path(directory) / "variants" / (shape + ".usda")
                build_scene(source, path, shape, "blue")
                scene = Usd.Stage.Open(str(path))
                obj = scene.GetPrimAtPath(body_path)
                self.assertEqual(scene.GetDefaultPrim().GetName(), "world")
                self.assertAlmostEqual(
                    UsdPhysics.MassAPI(obj).GetMassAttr().Get(), 0.01
                )
                active = [
                    p for p in Usd.PrimRange(obj) if p.HasAPI(UsdPhysics.CollisionAPI)
                ]
                self.assertEqual(len(active), 1)
                self.assertEqual(
                    UsdShade.MaterialBindingAPI(active[0])
                    .ComputeBoundMaterial("physics")[0]
                    .GetPath(),
                    material.GetPath(),
                )
                visual = (
                    scene.GetPrimAtPath(old_visual) if shape == "cube" else active[0]
                )
                self.assertEqual(visual.GetTypeName(), type_name)
                self.assertEqual(
                    UsdShade.MaterialBindingAPI(visual)
                    .ComputeBoundMaterial()[0]
                    .GetPrim()
                    .GetName(),
                    "VariantColor",
                )
                with self.assertRaises(FileExistsError):
                    build_scene(source, path, shape, "blue")
            self.assertEqual(original, source.read_bytes())
