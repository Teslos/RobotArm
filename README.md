# RobotArm — Meca500 R3 Isaac Sim

Isaac Sim 4.5 simulation of the **Mecademic Meca500 R3** 6-DOF desktop robot performing a busbar scan.

## Requirements

- NVIDIA Isaac Sim 4.5
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) with the `RobotArm` environment

## Setup

Clone the repo and activate the environment:

```powershell
git clone https://github.com/Teslos/RobotArm.git
cd RobotArm
micromamba activate RobotArm
```

Run the tests to verify the environment:

```powershell
micromamba run -n RobotArm python -m pytest tests/ --tb=short -q
```

## Running the simulation

### Demo mode — move arm above busbar centre

```powershell
micromamba run -n RobotArm python scripts/start_sim.py --demo
```

### RMPFlow mode — motion-planned approach to busbar

```powershell
micromamba run -n RobotArm python scripts/start_sim.py --rmpflow
```

### Scan mode — approach then sweep along busbar

```powershell
# Default: linear Y sweep (252 mm)
micromamba run -n RobotArm python scripts/start_sim.py --scan

# Headless (no GUI, faster)
micromamba run -n RobotArm python scripts/start_sim.py --scan --headless

# Keep window open after sweep finishes
micromamba run -n RobotArm python scripts/start_sim.py --scan --steps 0
```

### Common flags

| Flag | Description |
|---|---|
| `--headless` | Disable GUI rendering |
| `--steps N` | Stop after N physics steps (0 = run until Ctrl+C) |
| `--render-interval N` | Render every N steps (default 1). E.g. `4` gives ~15 FPS visual |

## Arbitrary scan paths

Generate a path file, then pass it to the sim:

```powershell
# Built-in shapes: zigzag, circle, spiral
micromamba run -n RobotArm python scripts/make_scan_path.py --shape spiral
micromamba run -n RobotArm python scripts/make_scan_path.py --shape zigzag
micromamba run -n RobotArm python scripts/make_scan_path.py --shape circle

# Run with the generated path
micromamba run -n RobotArm python scripts/start_sim.py --scan --scan-path results/scan_path.npy
```

Custom paths: save any `(N, 3)` numpy array of XYZ positions (in metres) as a `.npy` file and pass it via `--scan-path`.

```python
import numpy as np
waypoints = np.array([[0.136, y, 0.162] for y in np.linspace(-0.1, 0.1, 50)])
np.save("results/my_path.npy", waypoints)
```

## IK workspace grid search

Run a reachability survey over the busbar region and save results:

```powershell
micromamba run -n RobotArm python scripts/map_workspace.py
micromamba run -n RobotArm python scripts/map_workspace.py --nx 10 --ny 10 --nz 6
micromamba run -n RobotArm python scripts/map_workspace.py --output results/my_search.npz
```

Last result: `results/workspace.npz` — 202/320 reachable (63.1%), 8×8×5 grid.

## Visualisation

In scan mode the viewport shows:
- **Orange line** — planned EE path (IK targets)
- **Cyan trail** — actual EE path as the arm executes the trajectory

Camera is set to a side view (eye at X=+0.5 m) so the Y sweep appears as left-right motion.

## Project layout

```
exts/robot_arm/          # Isaac Sim extension (physics, control, sensors)
assets/mecademic_description/
  urdf/meca500r3.urdf    # Robot URDF
  meshes/                # .dae visual + .stl collision meshes
  rmpflow/               # LULA kinematic chain + RMPFlow config
scripts/
  start_sim.py           # Main sim launcher
  make_scan_path.py      # Scan path generator (zigzag / circle / spiral)
  map_workspace.py       # IK grid search → results/workspace.npz
results/
  workspace.npz          # Reachability map
  scan_path.npy          # Last generated scan path
config/robot_arm.yaml    # Default physics/control parameters
docs/specs.md            # Full design specification
tests/                   # Unit tests (no Isaac Sim required)
```

## Robot spec

**Mecademic Meca500 R3** — 6-DOF desktop robot

| Joint | Limit |
|---|---|
| J1 | ±175° |
| J2 | −70° → +90° |
| J3 | −135° → +70° |
| J4 | ±170° |
| J5 | ±115° |
| J6 | ±180° |
