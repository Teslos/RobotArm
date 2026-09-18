"""
Metrology stack for the INT_Monitor measurement rig.

This subpackage holds the reusable half of the scanning scripts that used to
live in ``C:\\EmpaDaten\\INT_Monitor\\py``: the instrument TCP protocol, the scan
geometry, a safety-checked robot facade, point logging and the FFT export.
The runnable entry points are in ``scripts/``.

Units are the robot's native **millimetres and degrees**, not the metres and
radians used by the Isaac Sim side of this repo.
"""
from .fft_export import (
    DEFAULT_VERSIONS,
    FftRow,
    build_rows,
    collect_fft_files,
    parse_coordinates,
    write_csv,
    write_xlsx,
)
from .instrument import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    END_TOKEN,
    AxisPair,
    InstrumentDisconnected,
    InstrumentError,
    InstrumentLink,
    InstrumentTimeout,
    serve,
)
from .limits import (
    DEFAULT_WORKSPACE,
    HARD_JOINT_LIMITS_DEG,
    SOFT_JOINT_LIMITS_DEG,
    LimitViolation,
    WorkspaceBox,
    check_joint_limits,
    check_pose,
    check_waypoints,
)
from .patterns import ScanPath, attach_orientation, line_scan, raster_grid
from .recording import PointLog, PointRecord, point_filename, write_point_file
from .robot_session import (
    DEFAULT_ROBOT_IP,
    RobotSession,
    VelocityLimits,
    connect_robot,
    normalize_pose,
)
from .scan import ScanResult, ScanSettings, run_scan

__all__ = [
    # instrument
    "AxisPair",
    "InstrumentLink",
    "InstrumentError",
    "InstrumentDisconnected",
    "InstrumentTimeout",
    "serve",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "END_TOKEN",
    # limits
    "WorkspaceBox",
    "LimitViolation",
    "DEFAULT_WORKSPACE",
    "HARD_JOINT_LIMITS_DEG",
    "SOFT_JOINT_LIMITS_DEG",
    "check_joint_limits",
    "check_pose",
    "check_waypoints",
    # patterns
    "ScanPath",
    "line_scan",
    "raster_grid",
    "attach_orientation",
    # robot
    "RobotSession",
    "VelocityLimits",
    "connect_robot",
    "normalize_pose",
    "DEFAULT_ROBOT_IP",
    # recording
    "PointLog",
    "PointRecord",
    "point_filename",
    "write_point_file",
    # scan
    "run_scan",
    "ScanSettings",
    "ScanResult",
    # fft export
    "FftRow",
    "parse_coordinates",
    "collect_fft_files",
    "build_rows",
    "write_csv",
    "write_xlsx",
    "DEFAULT_VERSIONS",
]
