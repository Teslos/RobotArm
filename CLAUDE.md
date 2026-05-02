# RobotArm — Isaac Sim Project

## Environment
- Python manager: **micromamba**, env: **RobotArm**
- Run commands: `micromamba run -n RobotArm python ...`
- Run tests: `micromamba run -n RobotArm python -m pytest tests/ --tb=short -q`

## Project layout
```
exts/robot_arm/     # Main package (Isaac Sim extension)
  config.py         # Typed dataclasses for all physics/control parameters
  world.py          # build_world(): TGS solver, Z-up, collision groups
  articulation.py   # configure_articulation(): drives, limits, CCD, offsets
  sensors.py        # LaggedSensor / LaggedCamera / LaggedLidar
  safeguards.py     # SafeguardManager: teleport guard + drift re-centering
  controller.py     # RobotArmController: RMPFlow + stall monitor
config/robot_arm.yaml   # YAML mirror of default config values
docs/specs.md           # Full design specification
tests/                  # Unit tests (no Isaac Sim required — omni/pxr stubbed)
```

## Key design decisions (from docs/specs.md)
- **Solver:** TGS at 60 Hz, 1:1 physics/RMPFlow sync
- **Articulation:** `fixed_base=True`, position drives, hard joint limits
- **Damping:** Critical damping `D = 2√K` per joint
- **CCD:** Enabled on `end_effector` and `sensor_arm` links only
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

## Physics parameters (defaults)
| Parameter | Value |
|---|---|
| `joint_friction` | 0.05 |
| `joint_damping` | 1.0 |
| `stiffness` | 400.0 |
| `rmpflow_timeout_s` | 5.0 |
| `stall_velocity_threshold` | 1e-3 |
