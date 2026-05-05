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
URDF_PATH  = os.path.join(ASSETS_DIR, "mecademic_description", "urdf", "meca500r3.urdf")
ROBOT_USD  = os.path.join(ASSETS_DIR, "mecademic_description", "urdf", "meca500r3.usd")
BUSBAR_STL = os.path.join(ASSETS_DIR, "busbar", "busbar.stl")
BUSBAR_USD = os.path.join(ASSETS_DIR, "busbar", "busbar.usd")
MESH_DIR       = os.path.join(ASSETS_DIR, "mecademic_description", "meshes")
RMPFLOW_DIR    = os.path.join(ASSETS_DIR, "mecademic_description", "rmpflow")
ROBOT_DESC     = os.path.join(RMPFLOW_DIR, "robot_descriptor.yaml")
RMPFLOW_CONFIG = os.path.join(RMPFLOW_DIR, "rmpflow_config.yaml")

# Meca500 R3 joint targets (degrees) — best approach to busbar centre.
# Derived from IK grid search (results/workspace.npz): nearest reachable point
# to busbar centre [0.15, 0.0, 0.18] m → EE at [0.136, 0.036, 0.162] m.
# Joint limits: J1 ±175°, J2 -70→90°, J3 -135→70°, J4 ±170°, J5 ±115°, J6 ±180°.
DEMO_JOINT_DEG = [-1.5, 35.0, 18.3, 36.9, 112.4, 12.9]

# Busbar scan parameters
SCAN_LENGTH_M       = 0.252   # 252 mm scan along the busbar Y axis
SCAN_HEIGHT_M       = 0.162   # hover height from IK grid search
SCAN_STEPS_APPROACH = 600     # steps to reach centre hover (10 s at 60 Hz)
SCAN_STEPS_SWEEP    = 1200    # steps for each sweep leg (20 s at 60 Hz)


# ── Asset conversion helpers ─────────────────────────────────────────────────

def _patch_urdf(urdf_path: str) -> str:
    """
    Resolve package:// mesh URIs to absolute paths when mesh files exist in
    MESH_DIR, otherwise strip visual/collision elements so the import still
    works (joints-only mode).  Handles both .dae and .stl mesh references.
    """
    has_meshes = os.path.isdir(MESH_DIR) and any(
        f.endswith((".dae", ".stl")) for f in os.listdir(MESH_DIR)
    )

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if has_meshes:
        for mesh_elem in root.iter("mesh"):
            fn = mesh_elem.get("filename", "")
            if fn.startswith("package://"):
                basename = os.path.basename(fn)
                abs_path = os.path.join(MESH_DIR, basename)
                if os.path.exists(abs_path):
                    mesh_elem.set("filename", abs_path)
        print(f"[asset] Using meshes from {MESH_DIR}")
    else:
        for link in root.iter("link"):
            for tag in ("visual", "collision"):
                for elem in link.findall(tag):
                    link.remove(elem)
        print("[asset] No mesh files found — importing joint structure only.")
        print(f"        Place .dae/.stl files in {MESH_DIR} and delete {ROBOT_USD} to re-import.")

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


def _acquire_draw():
    """Return (draw_interface, carb) or (None, None) if debug draw is unavailable."""
    try:
        import carb
        import omni.debugdraw
        return omni.debugdraw.get_debug_draw_interface(), carb
    except Exception:
        return None, None


def _set_viewport_camera() -> None:
    """Point the default viewport at the arm + busbar from a 45-degree isometric angle.

    Eye at (0.55, -0.45, 0.45) looks toward the workspace centre (0.1, 0.0, 0.12),
    giving a clear view of the full arm and the busbar on the table.
    """
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        try:
            from omni.isaac.core.utils.viewports import set_camera_view  # type: ignore[no-redef]
        except ImportError:
            return  # viewport API not available — skip silently
    import numpy as np
    set_camera_view(
        eye=np.array([0.55, -0.45, 0.45]),
        target=np.array([0.10, 0.00, 0.12]),
        camera_prim_path="/OmniverseKit_Persp",
    )


def _build_and_run(cfg, steps: int, app, demo: bool = False, rmpflow: bool = False,
                   scan: bool = False, headless: bool = False, render_interval: int = 1) -> None:
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
        name="meca500r3",
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

    if not headless:
        _set_viewport_camera()

    configure_articulation(robot, cfg, ccd_link_names=tuple(sc.ccd_link_names))

    print(f"[sim] Robot DOF count : {robot.num_dof}")
    print(f"[sim] Robot DOF names : {robot.dof_names}")
    print(f"[sim] Busbar position : {sc.busbar_position}")

    controller = None
    scan_waypoints: list = []
    scan_step_thresholds: list = []
    _draw = None
    _carb = None
    _scan_prev_ee = [None]  # last EE Float3 for trail drawing
    _plan_lines: list = []  # static planned-path draw calls
    _trail_lines: list = []  # growing EE trail draw calls

    if demo:
        target_rad = list(np.deg2rad(DEMO_JOINT_DEG))
        initial_pos_deg = np.degrees(robot.get_joint_positions()).tolist()
        print(f"[demo] Initial joints (deg): {[round(v,1) for v in initial_pos_deg]}")
        set_joint_position_targets(robot, target_rad)
        if steps == 0:
            steps = 600  # 10 s at 60 Hz
        print(f"[demo] Target  joints (deg): {DEMO_JOINT_DEG}")
        print(f"[demo] Running {steps} steps — watch arm move above busbar centre.")

    elif rmpflow:
        from exts.robot_arm.controller import RobotArmController
        controller = RobotArmController(
            robot=robot,
            robot_description_path=ROBOT_DESC,
            rmpflow_config_path=RMPFLOW_CONFIG,
            urdf_path=URDF_PATH,
            end_effector_frame_name="meca_axis_6_link",
            cfg=cfg,
        )
        controller.reset()
        # Busbar half-height is 0.05 m (cube scale z=0.05); approach 10 cm above.
        busbar_top_z = sc.busbar_position[2] + 0.05
        target = np.array([sc.busbar_position[0], sc.busbar_position[1], busbar_top_z + 0.10])
        controller.set_end_effector_target(target)
        if steps == 0:
            steps = 1200  # 20 s at 60 Hz
        print(f"[rmpflow] EE target: {target.tolist()} m")
        print(f"[rmpflow] Running {steps} steps.")

    elif scan:
        from exts.robot_arm.controller import RobotArmController
        controller = RobotArmController(
            robot=robot,
            robot_description_path=ROBOT_DESC,
            rmpflow_config_path=RMPFLOW_CONFIG,
            urdf_path=URDF_PATH,
            end_effector_frame_name="meca_axis_6_link",
            cfg=cfg,
        )
        controller.reset()
        bx, by = sc.busbar_position[0], sc.busbar_position[1]
        half = SCAN_LENGTH_M / 2.0
        scan_waypoints = [
            np.array([bx, by,          SCAN_HEIGHT_M]),  # 1. hover above centre
            np.array([bx, by - half,   SCAN_HEIGHT_M]),  # 2. sweep start  (-Y)
            np.array([bx, by + half,   SCAN_HEIGHT_M]),  # 3. sweep end    (+Y)
        ]
        scan_waypoint_steps = [SCAN_STEPS_APPROACH, SCAN_STEPS_SWEEP, SCAN_STEPS_SWEEP]
        scan_wp_idx = [0]  # mutable so inner loop can update it
        scan_step_thresholds = [sum(scan_waypoint_steps[:i+1]) for i in range(len(scan_waypoint_steps))]
        controller.set_end_effector_target(scan_waypoints[0])
        steps = sum(scan_waypoint_steps)
        print(f"[scan] Busbar scan: {SCAN_LENGTH_M*1000:.0f} mm along Y axis at Z={SCAN_HEIGHT_M:.3f} m")
        for i, wp in enumerate(scan_waypoints):
            print(f"[scan]   WP{i}: {wp.tolist()}  ({scan_waypoint_steps[i]} steps)")
        print(f"[scan] Total: {steps} steps ({steps/60:.0f} s)")
        # ── debug visualisation ───────────────────────────────────────────
        # omni.debugdraw is IMMEDIATE MODE: lines are cleared after every
        # render frame, so they must be redrawn on each world.step(render=True).
        _draw, _carb = _acquire_draw()
        # Pre-build static line list for the planned path (orange)
        _plan_lines: list = []  # each entry: (p0, col, w, p1, col, w)
        if _draw is not None:
            for i in range(len(scan_waypoints) - 1):
                p0 = _carb.Float3(*scan_waypoints[i].tolist())
                p1 = _carb.Float3(*scan_waypoints[i + 1].tolist())
                _plan_lines.append((p0, 0xFFFF8800, 6.0, p1, 0xFFFF8800, 6.0))
            r = 0.008  # cross-hair arm length (m)
            for wp in scan_waypoints:
                x, y, z = wp.tolist()
                for dx, dy, dz in [(r,0,0),(0,r,0),(0,0,r)]:
                    _plan_lines.append((
                        _carb.Float3(x-dx, y-dy, z-dz), 0xFFFF8800, 4.0,
                        _carb.Float3(x+dx, y+dy, z+dz), 0xFFFF8800, 4.0,
                    ))
            print("[scan] Debug draw ready — orange plan + cyan trail will render each frame.")
        else:
            print("[scan] omni.debugdraw not available — viewport lines disabled.")
        _trail_lines: list = []    # grows as arm moves; redrawn every frame

    if headless:
        print("[sim] Headless mode — rendering disabled (saves GPU/CPU heat).")
    elif render_interval > 1:
        print(f"[sim] Rendering every {render_interval} steps (~{60 // render_interval} FPS visual).")

    print(f"[sim] Running {'indefinitely (Ctrl+C to stop)' if steps == 0 else str(steps) + ' steps'}...")

    # Cache UsdGeom import once for EE trail sampling
    if scan and _draw is not None:
        from pxr import UsdGeom as _UsdGeom
        _ee_prim = world.stage.GetPrimAtPath(f"{sc.robot_prim_path}/meca_axis_6_link")
    else:
        _UsdGeom = None
        _ee_prim = None

    step_count = 0
    try:
        while steps == 0 or step_count < steps:
            do_render = (not headless) and (step_count % render_interval == 0)

            # Issue debug draw calls BEFORE world.step so they land in this frame
            # (omni.debugdraw is immediate-mode: queue is flushed during render)
            if scan and _draw is not None and do_render:
                for args in _plan_lines:
                    _draw.draw_line(*args)
                for args in _trail_lines:
                    _draw.draw_line(*args)

            world.step(render=do_render)
            if controller is not None:
                controller.step()

            # Every 3 steps: sample EE position and append a trail segment
            if scan and _draw is not None and step_count % 3 == 0 and _ee_prim is not None:
                try:
                    if _ee_prim.IsValid():
                        t = _UsdGeom.XformCache().GetLocalToWorldTransform(
                            _ee_prim
                        ).ExtractTranslation()
                        cur = _carb.Float3(float(t[0]), float(t[1]), float(t[2]))
                        if _scan_prev_ee[0] is not None:
                            _trail_lines.append((
                                _scan_prev_ee[0], 0xFF00FFFF, 4.0,
                                cur,              0xFF00FFFF, 4.0,
                            ))
                        _scan_prev_ee[0] = cur
                except Exception:
                    pass
            # Advance scan waypoints on threshold crossings
            if scan and step_count in scan_step_thresholds:
                next_idx = scan_step_thresholds.index(step_count) + 1
                if next_idx < len(scan_waypoints):
                    scan_wp_idx[0] = next_idx
                    controller.set_end_effector_target(scan_waypoints[next_idx])
                    print(f"[scan] → WP{next_idx}: {scan_waypoints[next_idx].tolist()}")
            step_count += 1
    except KeyboardInterrupt:
        print(f"\n[sim] Stopped after {step_count} steps.")

    if demo:
        final_pos_deg = np.degrees(robot.get_joint_positions()).tolist()
        target_deg = DEMO_JOINT_DEG
        print(f"[demo] Final   joints (deg): {[round(v,1) for v in final_pos_deg]}")
        print(f"[demo] Target  joints (deg): {target_deg}")
        errors = [abs(f - t) for f, t in zip(final_pos_deg, target_deg)]
        print(f"[demo] Errors          (deg): {[round(e,2) for e in errors]}")
        moved = any(abs(f - i) > 1.0 for f, i in zip(final_pos_deg, initial_pos_deg))
        print(f"[demo] Arm moved: {'YES' if moved else 'NO — joints did not change'}")

    print("[sim] Done.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="mechArm 270-Pi + busbar Isaac Sim launcher")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=0,
                        help="Number of physics steps (0 = run until Ctrl+C)")
    parser.add_argument("--render-interval", type=int, default=1, metavar="N",
                        help="Render every N physics steps (default: 1 = every step). "
                             "E.g. --render-interval 4 gives ~15 FPS visual at 60 Hz physics. "
                             "Ignored in --headless mode.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="Move arm above busbar centre using direct joint targets")
    mode.add_argument("--rmpflow", action="store_true",
                      help="Move arm above busbar centre using RMPFlow (needs LULA config files)")
    mode.add_argument("--scan", action="store_true",
                      help=f"Approach busbar then sweep {SCAN_LENGTH_M*1000:.0f} mm along Y axis")
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

    _build_and_run(cfg, args.steps, app, demo=args.demo, rmpflow=args.rmpflow,
                   scan=args.scan, headless=args.headless, render_interval=args.render_interval)

    app.close()


if __name__ == "__main__":
    main()
