"""Unit tests for LaggedSensor — no Isaac Sim required."""
import pytest
from exts.robot_arm.sensors import LaggedSensor
from exts.robot_arm.config import RobotArmCfg, SensorCfg, RobotArmCfg


def _make_sensor(lag: int = 1):
    counter = {"n": 0}

    def read():
        counter["n"] += 1
        return counter["n"]

    cfg = RobotArmCfg()
    cfg.sensor = SensorCfg(render_lag_frames=lag)
    sensor = LaggedSensor(read_fn=read, lag_frames=lag)
    return sensor, counter


def test_data_is_none_before_buffer_warms():
    sensor, _ = _make_sensor(lag=1)
    assert sensor.data is None  # buffer not yet warm
    sensor.step()
    assert sensor.data is None  # still only 1 entry, need lag+1=2


def test_data_lags_by_one_frame():
    sensor, _ = _make_sensor(lag=1)
    sensor.step()   # buffer: [1]
    sensor.step()   # buffer: [1, 2]  → data exposes frame 1
    assert sensor.data == 1
    sensor.step()   # buffer: [2, 3]  → data exposes frame 2
    assert sensor.data == 2


def test_lag_two_frames():
    sensor, _ = _make_sensor(lag=2)
    for _ in range(2):
        sensor.step()
    assert sensor.data is None   # need lag+1 = 3 entries
    sensor.step()                # now have 3 entries
    assert sensor.data == 1


def test_buffer_does_not_grow_unbounded():
    sensor, _ = _make_sensor(lag=1)
    for _ in range(100):
        sensor.step()
    assert len(sensor._buffer) == 2   # maxlen = lag + 1
