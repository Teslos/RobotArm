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
import asyncio
import os
import sys
import xml.etree.ElementTree as ET

# ── repo root on path so exts.robot_arm is importable ──────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ASSETS_DIR   = os.path.join(REPO_ROOT, "assets")
URDF_PATH    = os.path.join(ASSETS_DIR, "mecharm_270", "mecharm_270.urdf")
ROBOT_USD    = os.path.join(ASSETS_DIR, "mecharm_270", "mecharm_270.usd")
BUSBAR_STL   = os.path.join(ASSETS_DIR, "busbar", "busbar.stl")
BUSBAR_USD   = os.path.join(ASSETS_DIR, "busbar", "busbar.usd")
MESH_DIR     = os.path.join(ASSETS_DIR, "mecharm_270", "meshes")


# ── Asset conversion helpers ────────────────────────────────────────────────

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
        print(f"        Place .dae files in {MESH_DIR} and re-run for full geometry.")

    patched = urdf_path.replace(".urdf", "_patched.urdf")
    tree.write(patched, xml_declaration=True, encoding="unicode")
    return patched


def _convert_urdf_to_usd(app) -> None:
    import omni.kit.commands

    os.makedirs(os.path.dirname(ROBOT_USD), exist_ok=True)

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints   = False   # keep full link hierarchy for CCD
    import_config.fix_base             = True    # spec §1.3
    import_config.import_inertia_tensor = True
    import_config.distance_scale       = 1.0     # URDF is in metres
    import_config.create_physics_scene = False   # world.py owns the physics scene
    import_config.convex_decomp        = False   # convex hull per spec §3.3
    import_config.self_collision       = False   # collision groups in world.py

    patched = _patch_urdf(URDF_PATH)

    status, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=patched,
        import_config=import_config,
        dest_path=ROBOT_USD,
    )

    if os.path.exists(patched):
        os.remove(patched)

    if not status:
        raise RuntimeError(f"URDF import failed. Check {URDF_PATH} and Isaac Sim logs.")

    print(f"[asset] Robot USD written to {ROBOT_USD}")


async def _convert_stl_to_usd_async() -> None:
    import omni.kit.asset_converter as converter

    os.makedirs(os.path.dirname(BUSBAR_USD), exist_ok=True)

    ctx  = converter.AssetConverterContext()
    ctx.ignore_materials = False
    task = converter.get_instance().create_converter_task(
        BUSBAR_STL, BUSBAR_USD, None, ctx
    )
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(f"STL conversion failed: {task.get_error_message()}")
    print(f"[asset] Busbar USD written to {BUSBAR_USD}")


# ── Scene builder (correct ordering: configure drives AFTER world.reset()) ──

def _build_and_run(cfg, steps: int, app) -> None:
    """
    Build the scene manually so configure_articulation is called AFTER
    world.reset() — Isaac Sim populates dof_names only during initialization.
    """
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom

    from exts.robot_arm.articulation import configure_articulation
    from exts.robot_arm.world import build_world

    sc = cfg.scene

    # 1. World (TGS solver, Z-up, collision groups)
    world = build_world(cfg)

    # 2. Robot USD reference
    add_reference_to_stage(usd_path=sc.robot_usd_path, prim_path=sc.robot_prim_path)
    robot = Articulation(
        prim_path=sc.robot_prim_path,
        name="mecharm_270",
        position=list(sc.robot_position),
    )
    world.scene.add(robot)

    # 3. Busbar as static reference (position via USD translate op)
    add_reference_to_stage(usd_path=sc.busbar_usd_path, prim_path=sc.busbar_prim_path)
    busbar_prim = world.stage.GetPrimAtPath(sc.busbar_prim_path)
    if not busbar_prim.IsValid():
        raise RuntimeError(
            f"Busbar prim not found at '{sc.busbar_prim_path}'. "
            "Check that busbar_usd_path points to a valid USD file."
        )
    xform = UsdGeom.Xformable(busbar_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*sc.busbar_position))

    # 4. Initialize — MUST precede configure_articulation (populates dof_names)
    world.reset()

    # 5. Apply joint drives / limits / CCD now that articulation is initialised
    configure_articulation(robot, cfg, ccd_link_names=tuple(sc.ccd_link_names))

    print(f"[sim] Robot DOF count : {robot.num_dof}")
    print(f"[sim] Robot DOF names : {robot.dof_names}")
    print(f"[sim] Busbar position : {sc.busbar_position}")
    print(f"[sim] Running {'indefinitely (Ctrl+C to stop)' if steps == 0 else str(steps) + ' steps'}...")

    # 6. Physics loop
    step_count = 0
    try:
        while steps == 0 or step_count < steps:
            world.step(render=True)
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
    args = parser.parse_args()

    # SimulationApp must be created before any omni.* imports
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

    from exts.robot_arm.config import RobotArmCfg
    cfg = RobotArmCfg()

    # Convert assets on first run
    if not os.path.exists(ROBOT_USD):
        print("[asset] Converting URDF -> USD (first run)...")
        _convert_urdf_to_usd(app)
    else:
        print(f"[asset] Robot USD found: {ROBOT_USD}")

    if not os.path.exists(BUSBAR_USD):
        print("[asset] Converting STL -> USD (first run)...")
        asyncio.get_event_loop().run_until_complete(_convert_stl_to_usd_async())
    else:
        print(f"[asset] Busbar USD found: {BUSBAR_USD}")

    _build_and_run(cfg, args.steps, app)

    app.close()


if __name__ == "__main__":
    main()
