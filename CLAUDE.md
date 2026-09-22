# RobotArm — Isaac Sim Project

## Environment
- **OS: Windows 11** — use PowerShell syntax, NOT Linux/bash utilities
  - Delete files: `Remove-Item path` (not `rm` or `del`)
  - File exists check: `Test-Path path`
  - Multi-line Python: write a `.py` script file, do NOT use `python -c "..."` with multi-line strings
  - Grep equivalent: use the `Grep` tool, not `grep` or `rg` bash commands
  - Shell: PowerShell (`pwsh`) — use `$env:VAR`, backtick for line continuation, `&&` works
- Python manager: **micromamba**, env: **RobotArm**
- Run commands: `micromamba run -n RobotArm python ...`
- Run tests: `micromamba run -n RobotArm python -m pytest tests/ --tb=short -q`

## Robot
**Mecademic Meca500 R3** — 6-DOF desktop robot
- URDF: `assets/mecademic_description/urdf/meca500r3.urdf`
- Meshes: `assets/mecademic_description/meshes/` (.dae visual + .stl collision)
- EE link: `meca_axis_6_link`
- Joint limits (deg): J1 ±175, J2 -70→90, J3 -135→70, J4 ±170, J5 ±115, J6 ±180

## Project layout
```
exts/robot_arm/     # Main package (Isaac Sim extension)
  config.py         # Typed dataclasses for all physics/control parameters
  world.py          # build_world(): TGS solver, Z-up, collision groups
  articulation.py   # configure_articulation(): drives, limits, CCD, offsets
                    #   PD gains via USD DriveAPI (Isaac Sim 4.5 compatible)
  sensors.py        # LaggedSensor / LaggedCamera / LaggedLidar
  safeguards.py     # SafeguardManager: teleport guard + drift re-centering
  controller.py     # RobotArmController: RMPFlow + stall monitor
  robots/meca500.py # load_meca500(): USD reference + articulation setup
  tasks/workspace_mapper.py  # WorkspaceMapper: IK grid search for reachability
                              #   WorkspaceMap: load/save/query results (.npz)
  metrology/        # Physical-robot metrology stack (mm/deg, no Isaac Sim)
    limits.py       #   WorkspaceBox + joint limits + pre-flight validators
    instrument.py   #   INT_Monitor TCP handshake (AxisPair, InstrumentLink)
    patterns.py     #   ScanPath geometry: line_scan(), raster_grid()
    robot_session.py#   RobotSession: safety-checked mecademicpy facade
    recording.py    #   PointRecord / PointLog streaming CSV
    scan.py         #   run_scan(): move -> trigger -> wait -> log loop
    fft_export.py   #   INT_Monitor FFT .txt -> CSV / XLSX
assets/mecademic_description/
  urdf/meca500r3.urdf         # Robot URDF
  meshes/                     # .dae visual + .stl collision meshes
  rmpflow/
    robot_descriptor.yaml     # LULA kinematic chain + collision spheres
    rmpflow_config.yaml       # RMPFlow tuning params + base geometry
scripts/
  start_sim.py      # Main sim launcher (--demo, --rmpflow, --headless flags)
  map_workspace.py  # IK grid search over busbar region → results/workspace.npz
  scan_raster.py    # 2-D raster scan + INT_Monitor sync (real robot)
  scan_line.py      # 1-D line scan + INT_Monitor sync (real robot)
  export_fft_to_excel.py  # Merge INT_Monitor FFT .txt files into a workbook
  instrument_probe.py  # Diagnose the INT_Monitor TCP link (no robot needed)
  instrument_sim.py    # Fake INT_Monitor: answers each trigger after N s
results/
  workspace.npz     # Last grid search: 202/320 reachable (63.1%), 8×8×5 grid
config/robot_arm.yaml   # YAML mirror of default config values
docs/specs.md           # Full design specification
docs/int_monitor_port.md  # INT_Monitor port notes + bug list for the originals
tests/                  # Unit tests (no Isaac Sim required -- omni/pxr stubbed)
```

## Running scripts
```bash
# Interactive GUI simulation (demo movement)
micromamba run -n RobotArm python scripts/start_sim.py --demo

# Headless simulation with RMPFlow
micromamba run -n RobotArm python scripts/start_sim.py --headless --rmpflow --steps 200

# IK workspace grid search (saves to results/workspace.npz)
micromamba run -n RobotArm python scripts/map_workspace.py
micromamba run -n RobotArm python scripts/map_workspace.py --nx 10 --ny 10 --nz 6
micromamba run -n RobotArm python scripts/map_workspace.py --output results/my_search.npz

# --- Physical robot + INT_Monitor (mm/deg, no Isaac Sim needed) ---
# Always dry-run a new scan geometry first: validates waypoints, never moves.
micromamba run -n RobotArm python scripts/scan_raster.py --dry-run --n-fast 10 --n-slow 10 --step-fast 2 --step-slow 2
micromamba run -n RobotArm python scripts/scan_raster.py --fast-axis y --slow-axis x --n-fast 55 --n-slow 45 --step-fast 2 --step-slow 2 --axis-pair xy
micromamba run -n RobotArm python scripts/scan_line.py --axis y --n-points 120 --step 1 --axis-pair yz

# A scan ends at its home pose with the drives still active (no re-homing next
# run). --home sets that pose (X Y Z, or X Y Z A B C); --disconnect-when-done
# restores the old deactivate-and-disconnect finish.
micromamba run -n RobotArm python scripts/scan_raster.py --home 190 0 188 --n-fast 10 --n-slow 10

# TCP link not working? Probe it without the robot (the scanners home the arm
# BEFORE they open the socket, so a robot fault looks like a network fault).
micromamba run -n RobotArm python scripts/instrument_probe.py --listen-only
micromamba run -n RobotArm python scripts/instrument_probe.py --handshake --points 3

# No instrument available? Rehearse a scan against a fake one. The simulator is
# the CLIENT (INT_Monitor's role), so it runs next to a real scan on the arm.
micromamba run -n RobotArm python scripts/instrument_sim.py --acquire-time 2.0
micromamba run -n RobotArm python scripts/instrument_sim.py --acquire-time 2.0 --jitter 0.3
# Fault injection: --silent-after N (scan should hit --response-timeout),
#                  --drop-after N   (scan should report a disconnect).

# Merge INT_Monitor FFT text files into a workbook (xlsx needs openpyxl)
micromamba run -n RobotArm python scripts/export_fft_to_excel.py --root "C:/EmpaDaten/Data_Folder/mess38"
micromamba run -n RobotArm python scripts/export_fft_to_excel.py --root "C:/EmpaDaten/Data_Folder/mess38" --format csv
```

## Key design decisions (from docs/specs.md)
- **Solver:** TGS at 60 Hz, 1:1 physics/RMPFlow sync
- **Articulation:** `fixed_base=True`, position drives, hard joint limits
- **Damping:** Critical damping `D = 2√K` per joint
- **CCD:** Enabled on `meca_axis_6_link` (EE) only
- **Contact offsets:** contact=0.02 m, rest=0.00 m
- **Sensor lag:** Intentional 1-frame render lag for camera/lidar (mimics real hardware)
- **World:** Z-Up
- **Gripper:** Kinematic probe only — no grasping physics

## Safeguards
| Guard | Trigger | Action |
|---|---|---|
| Teleport guard | `safe_teleport()` called | Zeros all velocities + efforts |
| Drift re-centering | Every 10,000 steps, drift > 0.1 mm | Silently snaps base to origin |
| Stall monitor | Velocity < threshold for > 5 s | Cancels RMPFlow goal |

## Isaac Sim 4.5 API notes
- `SingleArticulation` has no `set_gains()` — set PD gains via `UsdPhysics.DriveAPI.CreateStiffnessAttr/DampingAttr` directly on joint prims
- `PhysxJointAPI` has no `CreateDampingAttr` — use only `CreateJointFrictionAttr`
- `LulaKinematicsSolver.compute_inverse_kinematics(frame_name, target_position, ...)` — `frame_name` is now a required first positional arg (e.g. `"meca_axis_6_link"`)
- `compute_inverse_kinematics` returns `(np.ndarray, bool)` not `(ArticulationAction, bool)`
- Use `get_cspace_position_limits()` to get n_dof from the solver

## Physics parameters (defaults)
| Parameter | Value |
|---|---|
| `joint_friction` | 0.05 |
| `joint_damping` | 1.0 |
| `stiffness` | 400.0 |
| `rmpflow_timeout_s` | 5.0 |
| `stall_velocity_threshold` | 1e-3 |
