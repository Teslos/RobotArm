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
RMPFLOW_CONFIG       = os.path.join(RMPFLOW_DIR, "rmpflow_config.yaml")
SCAN_WORKSPACE_NPZ   = os.path.join(REPO_ROOT, "results", "scan_workspace.npz")

# Meca500 R3 joint targets (degrees) — best approach to busbar centre.
# Derived from focused IK grid search (10×10×6, 100% reachable): nearest point
# to hover target [0.190, 0.0, 0.188] m → EE at [0.180, -0.006, 0.180] m.
# Joint limits: J1 ±175°, J2 -70→90°, J3 -135→70°, J4 ±170°, J5 ±115°, J6 ±180°.
DEMO_JOINT_DEG = [-2.4, -2.0, 52.3, -2.2, -32.1, -37.6]

# Busbar scan parameters
SCAN_LENGTH_M       = 0.252   # 252 mm scan along the busbar Y axis
SCAN_HEIGHT_M       = 0.188   # hover height — busbar top 0.1075 m + 80 mm clearance
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


def _set_viewport_camera(scan: bool = False) -> None:
    """Point the default viewport camera.

    Default (iso): eye=(0.55, -0.45, 0.45) → workspace centre.
    Scan mode:     eye=(0.50,  0.00, 0.35) → busbar scan line, looking along +X
                   so the Y sweep appears as left-right motion in the viewport.
    """
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        try:
            from omni.isaac.core.utils.viewports import set_camera_view  # type: ignore[no-redef]
        except ImportError:
            return
    import numpy as np
    if scan:
        # Side view: camera at positive X, looking toward the scan line.
        # Y motion appears left-right; Z height clearly visible.
        set_camera_view(
            eye=np.array([0.50, 0.00, 0.35]),
            target=np.array([0.15, 0.00, 0.16]),
            camera_prim_path="/OmniverseKit_Persp",
        )
    else:
        set_camera_view(
            eye=np.array([0.55, -0.45, 0.45]),
            target=np.array([0.10, 0.00, 0.12]),
            camera_prim_path="/OmniverseKit_Persp",
        )


def _build_and_run(cfg, steps: int, app, demo: bool = False, rmpflow: bool = False,
                   scan: bool = False, headless: bool = False, render_interval: int = 1,
                   scan_path: str | None = None) -> None:
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
    # 90° Y rotation: swaps X↔Z extents so the wide face (100×404 mm) is horizontal
    xform.AddRotateYOp().Set(90.0)

    # Initialise articulation — dof_names populated after this call
    world.reset()

    if not headless:
        _set_viewport_camera(scan=scan)

    configure_articulation(robot, cfg, ccd_link_names=tuple(sc.ccd_link_names))

    print(f"[sim] Robot DOF count : {robot.num_dof}")
    print(f"[sim] Robot DOF names : {robot.dof_names}")
    print(f"[sim] Busbar position : {sc.busbar_position}")

    controller = None
    scan_waypoints: list = []       # unused in IK-planned scan, kept for compat
    scan_step_thresholds: list = []
    _draw = None
    _carb = None
    _scan_prev_ee = [None]           # last EE Float3 for trail drawing
    _plan_lines: list = []           # static planned-path draw calls (redrawn each frame)
    _trail_lines: list = []          # growing EE trail draw calls
    _scan_jnt_traj: np.ndarray     = np.empty((0, 6))  # (N, n_dof) scan IK trajectory
    _approach_jnt_traj: np.ndarray = np.empty((0, 6))  # Cartesian approach IK waypoints
    _scan_start_joints: np.ndarray = np.zeros(6)        # rest pose at scan start

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
        # After 90° Y rotation busbar Z half-extent = 0.02125 m (was X scale); approach 10 cm above.
        busbar_top_z = sc.busbar_position[2] + 0.02125
        target = np.array([sc.busbar_position[0], sc.busbar_position[1], busbar_top_z + 0.10])
        controller.set_end_effector_target(target)
        if steps == 0:
            steps = 1200  # 20 s at 60 Hz
        print(f"[rmpflow] EE target: {target.tolist()} m")
        print(f"[rmpflow] Running {steps} steps.")

    elif scan:
        # ── Initialise LULA IK solver ──────────────────────────────────────
        try:
            from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        except ImportError:
            from omni.isaac.motion_generation.lula import LulaKinematicsSolver  # type: ignore
        _lk = LulaKinematicsSolver(
            robot_description_path=ROBOT_DESC,
            urdf_path=URDF_PATH,
        )

        # ── Pre-settle to approach pose; read EE orientation for IK constraint ──
        # Settling first means _scan_start_joints is already at the approach config,
        # and the orientation quaternion locks the wrist during IK planning so
        # the solver stays on the same branch for all 40 scan waypoints.
        set_joint_position_targets(robot, list(np.deg2rad(DEMO_JOINT_DEG)))
        for _ in range(120):        # 2 s of physics to settle
            world.step(render=False)
        _scan_start_joints = robot.get_joint_positions().copy()

        _ee_prim_fk = world.stage.GetPrimAtPath(f"{sc.robot_prim_path}/meca_axis_6_link")
        xform_fk  = UsdGeom.XformCache().GetLocalToWorldTransform(_ee_prim_fk)
        rot_fk    = xform_fk.ExtractRotation()
        quat_fk   = rot_fk.GetQuat()
        trans_fk  = xform_fk.ExtractTranslation()
        # EE Cartesian position at the settled approach pose
        ee_at_demo = np.array([float(trans_fk[0]), float(trans_fk[1]), float(trans_fk[2])])
        # LULA orientation constraint — [w, x, y, z]
        approach_ori = np.array([
            quat_fk.real,
            quat_fk.imaginary[0],
            quat_fk.imaginary[1],
            quat_fk.imaginary[2],
        ])
        print(f"[scan] Approach EE: pos={np.round(ee_at_demo,3).tolist()}  "
              f"ori(wxyz)={np.round(approach_ori,3).tolist()}")

        # ── Build scan EE target path (file or default linear sweep) ─────────
        SCAN_X_M = 0.190   # directly above busbar centre (top-down scan)
        by   = sc.busbar_position[1]   # 0.0 m
        half = SCAN_LENGTH_M / 2.0     # 0.126 m

        if scan_path is not None:
            scan_ee_targets = np.load(scan_path)
            if scan_ee_targets.ndim != 2 or scan_ee_targets.shape[1] != 3:
                raise ValueError(
                    f"scan_path file must contain an (N, 3) array, "
                    f"got shape {scan_ee_targets.shape}"
                )
            N_SCAN = len(scan_ee_targets)
            print(f"[scan] Loaded {N_SCAN}-point path from {scan_path}")
        else:
            N_SCAN = 40
            y_values = np.linspace(by - half, by + half, N_SCAN)
            scan_ee_targets = np.array([
                [SCAN_X_M, y, SCAN_HEIGHT_M] for y in y_values
            ])
            print(f"[scan] Using default linear sweep: {N_SCAN} points "
                  f"Y={((by-half)*1000):.0f}→{((by+half)*1000):.0f} mm")

        def _ik_with_ori(tgt, ori, warm=None):
            """IK with orientation constraint, warm_start, and position-only fallback."""
            for try_ori in [True, False]:
                kwargs: dict = {"target_position": tgt}
                if try_ori and ori is not None:
                    kwargs["target_orientation"] = ori
                if warm is not None:
                    kwargs["warm_start"] = warm
                try:
                    res, ok = _lk.compute_inverse_kinematics("meca_axis_6_link", **kwargs)
                    jpos = res if isinstance(res, np.ndarray) else np.asarray(res.joint_positions)
                    if bool(ok):
                        return jpos, True
                except TypeError:
                    pass
            return None, False

        # Force LULA to stay on a single IK branch:
        # - clear default seeds so only warm_start is used
        # - limit to 1 CCD descent so it only follows the warm_start gradient
        _lk.set_default_cspace_seeds([])
        _lk.max_num_descents = 1
        REJECT_THRESH = np.radians(40)  # reject IK if any joint jumps > 40°

        scan_start_pos = scan_ee_targets[0]

        # ── Cartesian approach trajectory (computed FIRST so endpoint seeds the scan) ──
        # IK at N_APPROACH Cartesian points from demo EE → scan start EE.
        # warm_start follows the arm along the approach — reject branch-flipped solutions.
        N_APPROACH = 20
        app_pts = np.array([
            (1.0 - s) * ee_at_demo + s * scan_start_pos
            for s in np.linspace(0.0, 1.0, N_APPROACH)
        ])
        a_warm = _scan_start_joints.copy()
        _app_list: list = []
        n_app_ok = 0
        for pt in app_pts:
            jpos_a, ok_a = _ik_with_ori(pt, approach_ori, warm=a_warm)
            if ok_a and np.any(np.abs(jpos_a - a_warm) > REJECT_THRESH):
                ok_a = False  # branch flip — hold previous
            if ok_a:
                a_warm = jpos_a
                n_app_ok += 1
            else:
                jpos_a = a_warm.copy()
            _app_list.append(jpos_a.copy())
        _approach_jnt_traj = np.array(_app_list)
        print(f"[scan] Approach traj: {n_app_ok}/{N_APPROACH} IK solved  "
              f"({np.degrees(np.abs(_approach_jnt_traj[-1]-_approach_jnt_traj[0]).max()):.1f}° max span)")

        # ── Scan trajectory: warm_start from approach endpoint (arm at scan start) ──
        warm = _approach_jnt_traj[-1].copy()   # end of approach = start of scan
        print(f"[scan] Planning {N_SCAN}-point scan IK  seed=approach_end …")
        _scan_jnt_list: list = []
        n_ok = n_rejected = 0
        for tgt in scan_ee_targets:
            jpos, ok = _ik_with_ori(tgt, approach_ori, warm=warm)
            if ok and np.any(np.abs(jpos - warm) > REJECT_THRESH):
                ok = False   # branch flip — discard
                n_rejected += 1
            if ok:
                warm = jpos
                n_ok += 1
            else:
                jpos = warm.copy()
            _scan_jnt_list.append(jpos.copy())
        _scan_jnt_traj = np.array(_scan_jnt_list)
        print(f"[scan] Scan IK: {n_ok}/{N_SCAN} accepted  {n_rejected} rejected (branch flip)")

        # Validate trajectory continuity
        j_range_deg = np.degrees(_scan_jnt_traj.max(axis=0) - _scan_jnt_traj.min(axis=0))
        print(f"[scan] Joint range (deg): {np.round(j_range_deg, 1).tolist()}")
        print(f"[scan] J1 {np.degrees(_scan_jnt_traj[0,0]):.1f}°→{np.degrees(_scan_jnt_traj[-1,0]):.1f}°  "
              f"J4 {np.degrees(_scan_jnt_traj[0,3]):.1f}°→{np.degrees(_scan_jnt_traj[-1,3]):.1f}°")

        steps = SCAN_STEPS_APPROACH + SCAN_STEPS_SWEEP
        print(f"[scan] Approach {SCAN_STEPS_APPROACH} steps + Sweep {SCAN_STEPS_SWEEP} steps "
              f"= {steps} steps ({steps/60:.0f} s)")

        # ── debug visualisation (immediate-mode: redrawn every render frame) ──
        _draw, _carb = _acquire_draw()
        if _draw is not None:
            # Orange line through actual IK target positions
            for i in range(len(scan_ee_targets) - 1):
                p0 = _carb.Float3(*scan_ee_targets[i].tolist())
                p1 = _carb.Float3(*scan_ee_targets[i + 1].tolist())
                _plan_lines.append((p0, 0xFFFF8800, 5.0, p1, 0xFFFF8800, 5.0))
            r = 0.008
            for wp in [scan_ee_targets[0], scan_ee_targets[N_SCAN // 2], scan_ee_targets[-1]]:
                x, y, z = wp.tolist()
                for dx, dy, dz in [(r, 0, 0), (0, r, 0), (0, 0, r)]:
                    _plan_lines.append((
                        _carb.Float3(x-dx, y-dy, z-dz), 0xFFFF8800, 4.0,
                        _carb.Float3(x+dx, y+dy, z+dz), 0xFFFF8800, 4.0,
                    ))
            print("[scan] Debug draw ready — orange plan + cyan EE trail.")
        else:
            print("[scan] omni.debugdraw not available.")

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
                        xf = _UsdGeom.XformCache().GetLocalToWorldTransform(
                            _ee_prim
                        ).ExtractTranslation()
                        cur = _carb.Float3(float(xf[0]), float(xf[1]), float(xf[2]))
                        if _scan_prev_ee[0] is not None:
                            _trail_lines.append((
                                _scan_prev_ee[0], 0xFF00FFFF, 8.0,
                                cur,              0xFF00FFFF, 8.0,
                            ))
                        _scan_prev_ee[0] = cur
                except Exception:
                    pass
            # Joint trajectory interpolation with S-curve motion profiling
            if scan and len(_scan_jnt_traj) > 0:
                if step_count < SCAN_STEPS_APPROACH and len(_approach_jnt_traj) > 1:
                    t = float(step_count) / max(SCAN_STEPS_APPROACH, 1)
                    ts = 3.0 * t * t - 2.0 * t * t * t          # cubic ease-in/out
                    idx_f = ts * (len(_approach_jnt_traj) - 1)
                    lo = int(idx_f)
                    hi = min(lo + 1, len(_approach_jnt_traj) - 1)
                    alpha = idx_f - lo
                    tgt_joints = (
                        (1.0 - alpha) * _approach_jnt_traj[lo]
                        + alpha       * _approach_jnt_traj[hi]
                    )
                else:
                    sweep_t = min(
                        float(step_count - SCAN_STEPS_APPROACH) / max(SCAN_STEPS_SWEEP, 1),
                        1.0,
                    )
                    sweep_ts = 3.0 * sweep_t * sweep_t - 2.0 * sweep_t * sweep_t * sweep_t
                    idx_f = sweep_ts * (len(_scan_jnt_traj) - 1)
                    lo = int(idx_f)
                    hi = min(lo + 1, len(_scan_jnt_traj) - 1)
                    alpha = idx_f - lo
                    tgt_joints = (
                        (1.0 - alpha) * _scan_jnt_traj[lo]
                        + alpha       * _scan_jnt_traj[hi]
                    )
                set_joint_position_targets(robot, tgt_joints.tolist())
                # Log progress + actual EE position every ~5 s
                if step_count % 300 == 0:
                    phase = "approach" if step_count < SCAN_STEPS_APPROACH else "sweep"
                    pct = (min(step_count, SCAN_STEPS_APPROACH + SCAN_STEPS_SWEEP)
                           / (SCAN_STEPS_APPROACH + SCAN_STEPS_SWEEP) * 100)
                    ee_str = ""
                    if _ee_prim is not None:
                        try:
                            xf2 = _UsdGeom.XformCache().GetLocalToWorldTransform(
                                _ee_prim
                            ).ExtractTranslation()
                            ee_str = (f"  EE=({xf2[0]:.3f}, {xf2[1]:.3f}, {xf2[2]:.3f})")
                        except Exception:
                            pass
                    print(f"[scan] step {step_count}  phase={phase}  {pct:.0f}%{ee_str}")
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
    parser.add_argument("--scan-path", metavar="FILE.npy",
                        help="Path to an (N,3) .npy file of EE XYZ waypoints for --scan mode. "
                             "Default: linear Y sweep over the busbar.")
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
                   scan=args.scan, headless=args.headless, render_interval=args.render_interval,
                   scan_path=args.scan_path)

    app.close()


if __name__ == "__main__":
    main()
