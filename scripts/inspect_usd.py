"""Inspect joint prim types in the generated meca500r3.usd."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pxr import Usd, UsdPhysics

USD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "mecademic_description", "urdf", "meca500r3.usd")

stage = Usd.Stage.Open(USD_PATH)
print("=== Joint prims ===")
for prim in stage.TraverseAll():
    name = prim.GetName().lower()
    if "joint" in name:
        is_rev = prim.IsA(UsdPhysics.RevoluteJoint)
        has_drive = UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetPositionAttr().IsValid() if is_rev else False
        print(f"  path={prim.GetPath()}")
        print(f"    typeName={prim.GetTypeName()!r}  IsRevoluteJoint={is_rev}  hasAngularDrive={has_drive}")
