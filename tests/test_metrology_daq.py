"""Unit tests for the NI DAQ voltage reader — nidaqmx is faked."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest


class _FakeChannels(list):
    def add_ai_voltage_chan(self, channel, **kwargs):
        self.append((channel, kwargs))


class _FakeTiming:
    def cfg_samp_clk_timing(self, rate, **kwargs):
        self.rate = rate
        self.kwargs = kwargs


class _FakeTask:
    def __init__(self):
        self.ai_channels = _FakeChannels()
        self.timing = _FakeTiming()
        self.closed = False

    def read(self, number_of_samples_per_channel=None, timeout=None):
        n = len(self.ai_channels)
        if number_of_samples_per_channel is None:
            return 1.5 if n == 1 else [1.5] * n
        return [[float(i + 1)] * number_of_samples_per_channel for i in range(n)]

    def close(self):
        self.closed = True


@pytest.fixture
def daq(monkeypatch):
    fake = types.ModuleType("nidaqmx")
    fake.Task = _FakeTask
    constants = types.ModuleType("nidaqmx.constants")
    constants.TerminalConfiguration = types.SimpleNamespace(
        DEFAULT="default", RSE="rse", NRSE="nrse", DIFF="diff", PSEUDO_DIFF="pd",
    )
    constants.AcquisitionType = types.SimpleNamespace(FINITE="finite")
    system = types.ModuleType("nidaqmx.system")
    system.System = types.SimpleNamespace(
        local=lambda: types.SimpleNamespace(
            devices=[types.SimpleNamespace(name="Dev1")]
        )
    )
    fake.constants, fake.system = constants, system
    monkeypatch.setitem(sys.modules, "nidaqmx", fake)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", constants)
    monkeypatch.setitem(sys.modules, "nidaqmx.system", system)

    from exts.robot_arm.metrology import daq as module
    return module


def test_settings_reject_bad_range(daq):
    with pytest.raises(daq.DaqError):
        daq.DaqSettings(min_v=5, max_v=5)


def test_settings_reject_unknown_terminal(daq):
    with pytest.raises(daq.DaqError):
        daq.DaqSettings(terminal="bogus")


def test_list_devices(daq):
    assert daq.list_devices() == ["Dev1"]


def test_read_voltage_single_channel(daq):
    with daq.VoltageReader("Dev1/ai0") as reader:
        assert reader.read_voltage() == pytest.approx([1.5])


def test_read_mean_multi_channel(daq):
    with daq.VoltageReader(["Dev1/ai0", "Dev1/ai1"]) as reader:
        assert reader.read_mean(10) == pytest.approx([1.0, 2.0])
        assert reader.read_samples(5).shape == (2, 5)


def test_close_releases_task(daq):
    reader = daq.VoltageReader("Dev1/ai0").open()
    task = reader._task
    reader.close()
    assert task.closed and reader._task is None


def test_read_requires_open(daq):
    with pytest.raises(daq.DaqError):
        daq.VoltageReader("Dev1/ai0").read_voltage()


def test_requires_a_channel(daq):
    with pytest.raises(daq.DaqError):
        daq.VoltageReader([])
