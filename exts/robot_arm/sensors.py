"""Sensor wrappers with intentional 1-frame render lag (spec §4.1)."""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque

import numpy as np

from .config import RobotArmCfg


class LaggedSensor:
    """
    Buffers sensor output for N frames before exposing it.

    The "Think" phase reads data from T-1 while "Act" runs at T,
    matching real-world sensor processing latency.
    """

    def __init__(
        self,
        read_fn: Callable[[], Any],
        lag_frames: int = 1,
    ) -> None:
        self._read_fn = read_fn
        self._buffer: Deque[Any] = deque(maxlen=lag_frames + 1)
        self._lag_frames = lag_frames

    def step(self) -> None:
        """Call once per physics step to advance the internal buffer."""
        self._buffer.append(self._read_fn())

    @property
    def data(self) -> Any | None:
        """Returns the reading from `lag_frames` steps ago, or None if not yet warm."""
        if len(self._buffer) <= self._lag_frames:
            return None
        return self._buffer[0]


class LaggedCamera(LaggedSensor):
    """Camera with 1-frame lag. Pass an Isaac Sim Camera instance as `camera`."""

    def __init__(self, camera, cfg: RobotArmCfg | None = None) -> None:
        if cfg is None:
            cfg = RobotArmCfg()
        super().__init__(
            read_fn=camera.get_rgba,
            lag_frames=cfg.sensor.render_lag_frames,
        )
        self._camera = camera


class LaggedLidar(LaggedSensor):
    """Lidar with 1-frame lag. Pass an Isaac Sim Lidar instance as `lidar`."""

    def __init__(self, lidar, cfg: RobotArmCfg | None = None) -> None:
        if cfg is None:
            cfg = RobotArmCfg()
        super().__init__(
            read_fn=lidar.get_point_cloud_data,
            lag_frames=cfg.sensor.render_lag_frames,
        )
        self._lidar = lidar
