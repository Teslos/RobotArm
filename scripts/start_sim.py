#!/usr/bin/env python3
"""
Launch Isaac Sim, convert assets on first run, build the scene, and step physics.

First run converts:
  assets/mecharm_270/mecharm_270.urdf  ->  assets/mecharm_270/mecharm_270.usd
  assets/busbar/busbar.stl             ->  assets/busbar/busbar.usd

Mesh note: the URDF references .dae files at
  package://mycobot_description/urdf/mecharm_270_pi/*.dae
If you have them, place them in:
  assets/mecharm_270/meshes/
The script imports structure-only otherwise (joints work, no visual geometry).

Usage:
    micromamba run -n RobotArm python scripts/start_sim.py
    micromamba run -n RobotArm python scripts/start_sim.py --headless
    micromamba run -n RobotArm python scripts/start_sim.py --steps 500
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

# ── repo root on path so exts.robot_arm is importable ──────────────────────
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
URDF_PATH  = os.path.join(ASSETS_DIR, "mecharm_270", "mecharm_270.urdf")
ROBOT_USD  = os.path.join(ASSETS_DIR, "mecharm_270", "mecharm_270.usd")
BUSBAR_STL = os.path.join(ASSETS_DIR, "busbar", "busbar.stl")
BUSBAR_USD = os.path.join(ASSETS_DIR, "busbar", "busbar.usd")
MESH_DIR       = os.path.join(ASSETS_DIR, "mecharm_270", "meshes")
RMPFLOW_DIR    = os.path.join(ASSETS_DIR, "mecharm_270", "rmpflow")
ROBOT_DESC     = os.path.join(RMPFLOW_DIR, "robot_descriptor.yaml")
RMPFLOW_CONFIG = os.path.join(RMPFLOW_DIR, "rmpflow_config.yaml")

# mechArm 270-Pi "above busbar centre" joint targets (degrees).
# Busbar centre ≈ (0.206, 0.202, 0.10) m; target EE height ~0.20 m.
# Tune joint 1 (base swing) and joints 2-4 (arm elevation) by running with --demo.
DEMO_JOINT_DEG = [45.0, 60.0, -90.0, 30.0, 0.0, 0.0]


# ── Asset conversion helpers ─────────────────────────────────────────────────

def _patch_urdf(urdf_path: str) -> str:
    """
    Resolve package:// mesh paths to absolute paths if .dae files exist in
    MESH_DIR, otherwise strip visual/collision elements so import still works.
    Writes a patched URDF next to the original and returns its path.
    """
    has_meshes = os.path.isdir(MESH_DIR) and any(
        f.endswith(".dae") for f in os.listdir(MESH_DIR)
    )

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if has_meshes:
        for mesh_elem in root.iter("mesh"):
            fn = mesh_elem.get("filename", "")
            if fn.startswith("package://mycobot_description/urdf/mecharm_270_pi/"):
                basename = os.path.basename(fn)
                mesh_elem.set("filename", os.path.join(MESH_DIR, basename))
        print(f"[asset] Using meshes from {MESH_DIR}")
    else:
        for link in root.iter("link"):
            for tag in ("visual", "collision"):
                for elem in link.findall(tag):
                    link.remove(elem)
        print("[asset] No .dae meshes found — importing joint structure only.")
        print(f"        Place .dae files in {MESH_DIR} and delete {ROBOT_USD} to re-import.")

    patched = urdf_path.replace(".urdf", "_patched.urdf")
    tree.write(patched, xml_declaration=True, encoding="unicode")
    return patched


def _convert_urdf_to_usd(app) -> None:
    import omni.kit.app
    import omni.kit.commands

    # Extension was renamed in Isaac Sim 4.x
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
        if ext_manager.is_extension_enabled(ext_name) or ext_manager.set_extension_enabled_immediate(ext_name, True):
            break

    os.makedirs(os.path.dirname(ROBOT_USD), exist_ok=True)
    patched = _patch_urdf(URDF_PATH)

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints    = False
    import_config.fix_base              = True
    import_config.import_inertia_tensor = True
    import_config.distance_scale        = 1.0
    import_config.create_physics_scene  = False
    import_config.convex_decomp         = False
    import_config.self_collision        = False

    status, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=patched,
        import_config=import_config,
        dest_path=ROBOT_USD,
    )

    if os.path.exists(patched):
        os.remove(patched)

    if not status:
        raise RuntimeError("URDF import failed — check Isaac Sim logs.")

    print(f"[asset] Robot USD written to {ROBOT_USD}")


def _convert_stl_to_usd(_app) -> None:
    """
    Convert busbar.stl to USD by reading vertices directly and writing a
    UsdGeom.Mesh — no Kit asset converter needed, so no event-loop pumping
    and no timeout risk.  STL units are millimetres; we scale to metres.
    """
    import struct
    from pxr import Usd, UsdGeom, Vt, Gf

    os.makedirs(os.path.dirname(BUSBAR_USD), exist_ok=True)

    with open(BUSBAR_STL, "rb") as f:
        f.read(80)  # header
        n_tri = struct.unpack("<I", f.read(4))[0]
        pts, counts, indices = [], [], []
        for _ in range(n_tri):
            f.read(12)  # face normal
            base = len(pts)
            for v in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                pts.append(Gf.Vec3f(x * 0.001, y * 0.001, z * 0.001))
                indices.append(base + v)
            counts.append(3)
            f.read(2)  # attribute byte count

    stage = Usd.Stage.CreateNew(BUSBAR_USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/busbar")
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    stage.GetRootLayer().Save()
    print(f"[asset] Busbar USD written to {BUSBAR_USD} ({n_tri} triangles, mm→m scaled)")


def _write_placeholder_busbar_usd(usd_path: str) -> None:
    """Write a minimal USD file with a box mesh as a busbar stand-in."""
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    cube = UsdGeom.Cube.Define(stage, "/busbar")
    # Real busbar: 42.5 × 404 × 100 mm  (half-extents × 2 = cube size 2 default)
    cube.AddScaleOp().Set(Gf.Vec3f(0.02125, 0.202, 0.05))
    stage.GetRootLayer().Save()
    print(f"[asset] Placeholder busbar USD written to {usd_path}")


# ── Scene builder ────────────────────────────────────────────────────────────

def _get_usd_root_prim_name(usd_path: str) -> str:
    """Return the name of the first root-level prim in a USD file."""
    from pxr import Usd
    stage = Usd.Stage.Open(usd_path)
    default = stage.GetDefaultPrim()
    if default and default.IsValid():
        return default.GetName()
    # Fall back to the first child of the pseudo-root
    for prim in stage.GetPseudoRoot().GetChildren():
        return prim.GetName()
    raise RuntimeError(f"No root prims found in {usd_path}")


def _build_and_run(cfg, steps: int, app, demo: bool = False, rmpflow: bool = False) -> None:
    """
    Build the scene in the correct order:
      1. USD prim setup  (before world.reset)
      2. world.reset()   (initialises articulation, populates dof_names)
      3. configure_articulation  (requires dof_names — must come after reset)
    """
    import numpy as np
    from omni.isaac.core.articulations import Articulation
    from pxr import Gf, Sdf, Usd, UsdGeom

    from exts.robot_arm.articulation import configure_articulation, set_joint_position_targets
    from exts.robot_arm.world import build_world

    sc = cfg.scene

    world = build_world(cfg)

    # Discover the root prim name from the robot USD (URDF robot name may differ
    # from the file name, e.g. "firefighter").  We need to pass it explicitly
    # because the importer does not set defaultPrim metadata.
    robot_usd_root = _get_usd_root_prim_name(sc.robot_usd_path)
    robot_prim = world.stage.DefinePrim(sc.robot_prim_path)
    robot_prim.GetReferences().AddReference(sc.robot_usd_path, f"/{robot_usd_root}")

    robot = Articulation(
        prim_path=sc.robot_prim_path,
        name="mecharm_270",
        position=list(sc.robot_position),
    )
    world.scene.add(robot)

    # Busbar placeholder has /busbar as its root prim
    busbar_usd_root = _get_usd_root_prim_name(sc.busbar_usd_path)
    busbar_stage_prim = world.stage.DefinePrim(sc.busbar_prim_path)
    busbar_stage_prim.GetReferences().AddReference(sc.busbar_usd_path, f"/{busbar_usd_root}")
    busbar_prim = world.stage.GetPrimAtPath(sc.busbar_prim_path)
    if not busbar_prim.IsValid():
        raise RuntimeError(
            f"Busbar prim not found at '{sc.busbar_prim_path}'. "
            "Check that busbar_usd_path points to a valid USD file."
        )
    xform = UsdGeom.Xformable(busbar_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*sc.busbar_position))

    # Initialise articulation — dof_names populated after this call
    world.reset()

    configure_articulation(robot, cfg, ccd_link_names=tuple(sc.ccd_link_names))

    print(f"[sim] Robot DOF count : {robot.num_dof}")
    print(f"[sim] Robot DOF names : {robot.dof_names}")
    print(f"[sim] Busbar position : {sc.busbar_position}")

    controller = None

    if demo:
        target_rad = list(np.deg2rad(DEMO_JOINT_DEG))
        set_joint_position_targets(robot, target_rad)
        if steps == 0:
            steps = 600  # 10 s at 60 Hz
        print(f"[demo] Target joints (deg): {DEMO_JOINT_DEG}")
        print(f"[demo] Running {steps} steps — watch arm move above busbar centre.")

    elif rmpflow:
        from exts.robot_arm.controller import RobotArmController
        controller = RobotArmController(
            robot=robot,
            robot_description_path=ROBOT_DESC,
            rmpflow_config_path=RMPFLOW_CONFIG,
            urdf_path=URDF_PATH,
            end_effector_frame_name="link6",
            cfg=cfg,
        )
        controller.reset()
        busbar_cx = sc.busbar_position[0] + 0.02125 / 2
        busbar_cy = sc.busbar_position[1] + 0.202
        busbar_cz = sc.busbar_position[2] + 0.10 + 0.10  # 10 cm above top surface
        target = np.array([busbar_cx, busbar_cy, busbar_cz])
        controller.set_end_effector_target(target)
        if steps == 0:
            steps = 1200  # 20 s at 60 Hz
        print(f"[rmpflow] EE target: {target.tolist()} m")
        print(f"[rmpflow] Running {steps} steps.")

    print(f"[sim] Running {'indefinitely (Ctrl+C to stop)' if steps == 0 else str(steps) + ' steps'}...")

    step_count = 0
    try:
        while steps == 0 or step_count < steps:
            world.step(render=True)
            if controller is not None:
                controller.step()
            step_count += 1
    except KeyboardInterrupt:
        print(f"\n[sim] Stopped after {step_count} steps.")

    print("[sim] Done.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="mechArm 270-Pi + busbar Isaac Sim launcher")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=0,
                        help="Number of physics steps (0 = run until Ctrl+C)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="Move arm above busbar centre using direct joint targets")
    mode.add_argument("--rmpflow", action="store_true",
                      help="Move arm above busbar centre using RMPFlow (needs LULA config files)")
    args = parser.parse_args()

    # SimulationApp must be created before any omni.* imports
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

    from exts.robot_arm.config import RobotArmCfg
    cfg = RobotArmCfg()

    if not os.path.exists(ROBOT_USD):
        print("[asset] Converting URDF -> USD (first run)...")
        _convert_urdf_to_usd(app)
    else:
        print(f"[asset] Robot USD found: {ROBOT_USD}")

    if not os.path.exists(BUSBAR_USD):
        print("[asset] Converting STL -> USD (first run)...")
        _convert_stl_to_usd(app)
    else:
        print(f"[asset] Busbar USD found: {BUSBAR_USD}")

    _build_and_run(cfg, args.steps, app, demo=args.demo, rmpflow=args.rmpflow)

    app.close()


if __name__ == "__main__":
    main()
