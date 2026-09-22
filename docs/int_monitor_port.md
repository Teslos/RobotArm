# INT_Monitor scripts — port notes

The seven scripts in `C:\EmpaDaten\INT_Monitor\py` were re-implemented as three
CLIs in `scripts/` on top of a shared, tested package in
`exts/robot_arm/metrology/`. The originals are untouched.

Units throughout the metrology stack are **millimetres and degrees** (the
robot's native units), not the metres and radians used by the Isaac Sim side of
this repo.

## Where each original went

| Original | Replacement | Equivalent invocation |
|---|---|---|
| `Any Lines in Y Direaction Scanning.py` | `scripts/scan_raster.py` | `--fast-axis y --slow-axis x --n-fast 55 --n-slow 45 --step-fast 2 --step-slow 2 --axis-pair xy` |
| `Finall  Rectangular Mesurment.py` | `scripts/scan_raster.py` | `--fast-axis x --slow-axis y --n-fast 100 --n-slow 100 --step-fast 1 --step-slow 1 --axis-pair xy` |
| `RECTANGULAR _Pattern_Meca2 Wsafty.py` | `scripts/scan_raster.py` | `--n-fast 10 --n-slow 10 --step-fast 2 --step-slow 2 --hop-height 2 --settle 1.0` |
| `Width X And Z Measurment .py` | `scripts/scan_line.py` | `--axis x --n-points 22 --step 0.5 --axis-pair xz --settle 1.0 --verify-up 40` |
| `UPDATE mit SYNCRO.py` | `scripts/scan_line.py` | `--axis x --n-points 10 --step -5 --axis-pair xz --settle 0.5 --verify-up 20 --verify-down 20` |
| `Y and Z Measurment.py` | `scripts/scan_line.py` | `--axis y --n-points 120 --step 1 --axis-pair yz --settle 1.0 --verify-up 40` |
| `txt to excel.py` | `scripts/export_fft_to_excel.py` | `--root "C:/EmpaDaten/Data_Folder/mess38"` |

Both scanners take `--dry-run`, which builds and validates the waypoints, saves
them to a `.npy`, and exits without contacting the robot.

## Module layout

```
exts/robot_arm/metrology/
  limits.py          WorkspaceBox, joint limits, pre-flight validators
  instrument.py      AxisPair, InstrumentLink (the TCP handshake), serve()
  patterns.py        ScanPath, line_scan(), raster_grid()
  robot_session.py   RobotSession — safety-checked mecademicpy facade
  recording.py       PointRecord, PointLog (streaming CSV), legacy txt dumps
  scan.py            run_scan() — the move/trigger/wait/log loop
  fft_export.py      INT_Monitor FFT text files → CSV / XLSX
```

`scripts/` adds `exts/robot_arm` to `sys.path` and imports `metrology` as a
top-level package, deliberately bypassing `exts/robot_arm/__init__.py`, which
pulls in Isaac Sim. Driving the physical arm needs no simulator.

## Bugs found in the originals

### Crashes — these scripts could not complete a run as written

| # | File | Bug |
|---|---|---|
| 1 | `Width X And Z Measurment .py` | `start_pose = home_pose()` calls a list → `TypeError`. Same for `up_pose = start_pose()` and `down_pose = start_pose()`. |
| 2 | `Width X And Z Measurment .py` | `save_point()` references `OUTPUT_LINE_FOLDER`, whose definition is commented out → `NameError` whenever it is called. |
| 3 | `Width X And Z Measurment .py`, `Y and Z Measurment.py`, `UPDATE mit SYNCRO.py` | `robot.MovePose(pose)` passes a list; the SDK takes six positional arguments → `TypeError` on the first move. |
| 4 | `UPDATE mit SYNCRO.py` | `robot` is created inside the `try`; the `finally` references it, so a failure before that line raises `NameError` and hides the real error. |

### Motion correctness

| # | File | Bug |
|---|---|---|
| 5 | `RECTANGULAR _Pattern_Meca2 Wsafty.py` | Steps `10 × 2.0 mm = 20 mm` along X but rewinds `MoveLinRelWrf(-18, …)`. Every line starts 2 mm further out — **20 mm skew over a 10-line scan**. |
| 6 | `Finall  Rectangular Mesurment.py` | The rewind is a literal `-100 mm` that only matches `N_points_per_line * d_x` for the values checked in; the comment next to it still claims 20 mm. Change either constant and the grid shears. |
| 7 | `RECTANGULAR _Pattern_Meca2 Wsafty.py` | Recorded coordinates come from a counter (`point_idx % 10 * d_x`), not from `GetRtCartPos()`, so the data file shows a perfect grid while the arm drifts (bug 5). |
| 8 | Both rectangular scripts | The "safety hop" moves up by `safety_z` and straight back down with no travel in between, so it protects nothing. In `Finall  Rectangular Mesurment.py`, `safety_z = 0` makes it a literal no-op. |
| 9 | `Width X And Z Measurment .py`, `Y and Z Measurment.py` | `STEP` is negative *and* applied as `target = start - i * STEP`, so the arm travels opposite to what the constant reads. |
| 10 | All scan scripts | No workspace or joint-limit validation anywhere. A mistyped step size drives the arm into the fixture. |
| 11 | `Finall  Rectangular Mesurment.py` | The first point of each line is one step past the line start, so the recorded origin is never measured (inconsistent with the Y-direction scanner, which does measure it). |

### Protocol and control flow

| # | File | Bug |
|---|---|---|
| 12 | All scan scripts | `recv()` blocks with no timeout. A missed `done` hangs the scan forever with the arm powered and stationary. |
| 13 | All scan scripts | Bytes arriving after the `done` token are discarded; the next point relies on a pre-trigger purge to clean up. Two `done`s in one packet lose one. |
| 14 | `Finall  Rectangular Mesurment.py` | Announces axis command `yz` but streams `x… y…` labels. |
| 15 | `Width X And Z Measurment .py` | Sends the literal string `output_line` as the axis command. |
| 16 | `UPDATE mit SYNCRO.py` | `send_tcp()` never sends the trigger, unlike the other five scripts — the instrument is told where the arm is but never told to acquire. |
| 17 | All scan scripts | `while True: accept()` never exits, so a finished scan silently re-runs for every new client connection, from whatever pose the arm happens to be in. |
| 18 | `Finall  Rectangular Mesurment.py` | `execute_grid_scan()` returns `False` on failure; the caller ignores it and prints `🎉 Complete!` regardless. |
| 19 | `Finall  Rectangular Mesurment.py` | That message also hard-codes "100 total points" for a 100×100 = 10 000-point scan. |
| 20 | `RECTANGULAR _Pattern_Meca2 Wsafty.py`, `Width X And Z Measurment .py` | Bare `finally: robot.DeactivateRobot()`; if `Connect` failed, the deactivate raises and buries the original exception. |
| 21 | `Any Lines in Y Direaction Scanning.py` | `wait_for_end_token()` returns `False` on disconnect but a float otherwise, and the caller distinguishes them with `is False` — it works, but one refactor away from silently treating a disconnect as a 0-second acquisition. |
| 22 | `Any Lines in Y Direaction Scanning.py` | Each line moves to `line_start_pose` and then immediately issues the same pose again as loop iteration `i = 0`. |
| 23 | `Finall  Rectangular Mesurment.py` | Per-point files are `data_{n:02d}.txt` for a 10 000-point scan, so `data_100.txt` sorts before `data_99.txt`. |
| 31 | All scan scripts | **Every acquisition is attributed to the wrong point.** Connection setup fires a priming trigger, which starts a real acquisition, but the code only clears the socket for 50 ms before scanning. The priming `done` is therefore still in flight when the first point is triggered and is consumed as *that* point's response. Each point from then on reports the previous point's acquisition and, worse, the arm moves to the next waypoint while the instrument is still measuring the current one. The last point's `done` is never awaited at all. Confirmed on the wire with `scripts/instrument_probe.py`: against an instrument with a 0.3 s acquisition, point 1 returned in 0.063 s. |

### `txt to excel.py`

| # | Bug |
|---|---|
| 24 | `get_y_value` uses `[yY]([\d.]+)`, which cannot match a minus sign. Every folder on the negative side of the origin reads as `0.0`, so rows sort wrongly — silently, because "not found" also returns `0.0`. |
| 25 | `[\d.]+` matches `12.5.3`; the following `float()` raises `ValueError` from inside `list.sort`, aborting the entire export over one bad folder name. |
| 26 | `get_z_value`'s docstring says it extracts Z, its regex matches `[xX]`, and nothing calls it. |
| 27 | Measurement root, version list, glob depth, encoding, data-line index and output name are all hard-coded. |
| 28 | The coordinate label is uppercased into one opaque string, so the numeric position is lost in the output sheet. |
| 29 | `glob.glob` order is arbitrary and is only partially re-sorted afterwards. |
| 30 | Requires `pandas` purely to write a sheet, which neither the `RobotArm` env nor the system Python has installed. |

## What the replacements do differently

* **Absolute waypoints.** Every target is computed from one recorded origin, so
  no move depends on the accumulated result of previous moves (bugs 5–7).
* **Pre-flight validation.** The whole waypoint list, including safety-hop
  poses, is bounds-checked before the first move command is sent (bug 10).
* **Measured, not assumed.** Logged coordinates are read back from the robot
  (bug 7).
* **Bounded waits.** Per-point response timeouts, with `InstrumentTimeout` and
  `InstrumentDisconnected` distinguished (bugs 12, 21).
* **One trigger, one `done`.** `InstrumentLink.initialize()` waits for the
  priming trigger's response instead of draining for 50 ms, so no point is ever
  handed the previous acquisition's token (bug 31). Pass `priming_timeout_s=0`
  for an instrument that does not answer the priming trigger.
* **One connection, then exit** by default; `max_connections=0` restores the old
  unbounded behaviour (bug 17).
* **Failures propagate.** An instrument error aborts the scan and is reported as
  a failure (bug 18).
* **Streaming CSV log**, flushed per point, so an aborted scan keeps everything
  measured so far — plus optional legacy per-point text files with correct
  zero-padding (bug 23).
* **Real safety hop**: up at the end of a line, across at height, down at the
  next line start (bug 8). `--serpentine` avoids the rewind entirely.

## When no data reaches Python

`scripts/instrument_probe.py` opens the same TCP server as the scanners, but
with no robot and no Isaac Sim, and prints every byte in both directions with
timestamps.

```powershell
# Does INT_Monitor connect at all, and does it send anything unprompted?
micromamba run -n RobotArm python scripts/instrument_probe.py --listen-only

# Full handshake plus three fake points - the exact exchange run_scan performs.
micromamba run -n RobotArm python scripts/instrument_probe.py --handshake --points 3
```

## Rehearsing a scan without the instrument

`scripts/instrument_sim.py` is the probe's mirror image: the probe plays the PC
side (it listens, as the scanners do), the simulator plays INT_Monitor (it
dials in) and answers every trigger with `done` after a fixed delay.  Run it
beside a real scan and the arm moves for real while the measurement is faked,
which is how a raster's wall-clock cost is estimated before committing to it.

```powershell
# Terminal 1 - the real scan, real arm.
micromamba run -n RobotArm python scripts/scan_raster.py --n-fast 10 --n-slow 10

# Terminal 2 - stands in for INT_Monitor, 2 s per acquisition.
micromamba run -n RobotArm python scripts/instrument_sim.py --acquire-time 2.0
```

It retries the connection for `--retry-for` seconds, because the scanners bind
only after homing.  `--silent-after N` and `--drop-after N` inject the two
failures worth rehearsing: a stalled instrument (the scan must raise
`InstrumentTimeout` at `--response-timeout`) and one that hangs up mid-scan
(`InstrumentDisconnected`).  Pointing it at the probe instead of a scan
exercises the whole protocol with no hardware at all:

```powershell
micromamba run -n RobotArm python scripts/instrument_probe.py --handshake --points 3
micromamba run -n RobotArm python scripts/instrument_sim.py --acquire-time 0.4 --points 4
```

The scanners connect and home the robot **before** they open the socket, so a
robot fault looks exactly like a network fault: the server never starts
listening and INT_Monitor's connect is refused. Check with

```powershell
Get-NetTCPConnection -LocalPort 6340 -ErrorAction SilentlyContinue
```

If nothing is listening, Python is not at the accept yet. If the probe binds
and INT_Monitor still never connects, check that it targets `localhost:6340`
and that it is set to connect *now* — it opens the connection only when a
measurement starts. If it targets a LAN address rather than loopback, the
default `localhost` bind will not receive it; use `--host 0.0.0.0` (and
`--host` on the scanners).

## Running the tests

```powershell
micromamba run -n RobotArm python -m pytest tests/ --tb=short -q
```

`tests/test_metrology_*.py` need no robot, no Isaac Sim and no instrument. The
socket tests use a real `socket.socketpair()`, so timeouts and partial reads
behave as they do on the wire. The two `.xlsx` writer tests skip unless
`openpyxl` is installed.
