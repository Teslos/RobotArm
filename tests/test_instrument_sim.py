"""
Tests for the INT_Monitor simulator.

The simulator is driven through a real ``socket.socketpair()`` with the real
:class:`InstrumentLink` on the PC side, so what is exercised here is the actual
wire protocol rather than a description of it.  Sleeps are stubbed out, so a
"2 second" acquisition costs nothing.
"""
from __future__ import annotations

import importlib.util
import os
import random
import select
import socket
import threading
import time

import pytest

from exts.robot_arm.metrology.instrument import (
    AxisPair,
    InstrumentDisconnected,
    InstrumentLink,
    InstrumentTimeout,
)

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "instrument_sim.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("instrument_sim", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


instrument_sim = _load_module()
SimulatedInstrument = instrument_sim.SimulatedInstrument


@pytest.fixture
def pair():
    left, right = socket.socketpair()
    try:
        yield left, right
    finally:
        left.close()
        right.close()


def make_sim(conn, **kwargs):
    """A simulator whose acquisition sleep is instant but recorded."""
    slept = []
    kwargs.setdefault("sleep", slept.append)
    kwargs.setdefault("log", lambda _message: None)
    sim = SimulatedInstrument(conn, **kwargs)
    sim.slept = slept
    return sim


def make_link(conn, **kwargs):
    kwargs.setdefault("command_gap_s", 0.0)
    kwargs.setdefault("response_timeout_s", 2.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return InstrumentLink(conn, **kwargs)


def run_in_thread(sim):
    thread = threading.Thread(target=sim.run, daemon=True)
    thread.start()
    return thread


# ── The happy path ───────────────────────────────────────────────────────────

def test_simulator_answers_the_priming_trigger_and_every_point(pair):
    """initialize() waits for a 'done'; the sim must answer it like any other."""
    pc, instrument = pair
    sim = make_sim(instrument, acquire_s=2.0)
    run_in_thread(sim)

    link = make_link(pc, axis_pair=AxisPair.XY)
    link.initialize(0.0)

    for _ in range(3):
        link.measure_point("x190.0 y0.0")

    pc.close()
    assert sim.slept == [2.0, 2.0, 2.0, 2.0]  # priming + 3 points


def test_acquire_time_is_what_the_scan_measures(pair):
    """The sim's sleep is the scan's acquire_s, so a scan can be timed."""
    pc, instrument = pair
    sim = make_sim(instrument, acquire_s=0.25, sleep=time.sleep)
    run_in_thread(sim)

    link = make_link(pc)
    link.initialize(0.0)
    elapsed = link.measure_point("x1.0 y2.0")

    pc.close()
    assert 0.2 <= elapsed < 1.0


def test_axis_command_is_recognised_not_treated_as_a_label(pair):
    pc, instrument = pair
    sim = make_sim(instrument)
    run_in_thread(sim)

    link = make_link(pc, axis_pair=AxisPair.YZ)
    link.initialize(0.0)
    assert sim.label == ""  # 'yz' was the axis command, not a point

    link.measure_point("y1.0 z2.0")
    pc.close()
    assert sim.label == "y1.0 z2.0"


def test_a_label_split_across_packets_is_reassembled(pair):
    pc, instrument = pair
    sim = make_sim(instrument)
    run_in_thread(sim)

    pc.sendall(b"x12.3 y-4")
    pc.sendall(b"5.6\r\n")
    link = make_link(pc)
    link.trigger()
    link.wait_for_done()

    pc.close()
    assert sim.label == "x12.3 y-45.6"


def test_banner_is_sent_on_connect_and_drained_by_initialize(pair):
    pc, instrument = pair
    sim = make_sim(instrument, banner="INT_Monitor v1.2")
    run_in_thread(sim)

    # The real link waits `settle_s` before draining; here the sleeps are
    # stubbed out, so wait for the banner to actually land instead of racing
    # the simulator thread.
    ready, _, _ = select.select([pc], [], [], 2.0)
    assert ready, "the simulator never sent its banner"

    link = make_link(pc)
    banner = link.initialize(0.0)

    pc.close()
    assert b"INT_Monitor v1.2" in banner


# ── Stop conditions ──────────────────────────────────────────────────────────

def test_points_limit_stops_the_simulator(pair):
    pc, instrument = pair
    sim = make_sim(instrument, max_points=2)
    thread = run_in_thread(sim)

    link = make_link(pc)
    link.trigger()
    link.wait_for_done()
    link.measure_point("x1.0 y1.0")

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert sim.answered == 2


# ── Fault injection ──────────────────────────────────────────────────────────

def test_silent_after_makes_the_scan_time_out(pair):
    """--silent-after exercises the scan's --response-timeout path."""
    pc, instrument = pair
    sim = make_sim(instrument, silent_after=1)
    run_in_thread(sim)

    link = make_link(pc, response_timeout_s=0.2)
    link.trigger()
    link.wait_for_done()  # answered

    with pytest.raises(InstrumentTimeout):
        link.measure_point("x1.0 y1.0")
    pc.close()


def test_drop_after_makes_the_scan_see_a_disconnect(pair):
    """--drop-after exercises the scan's disconnect path."""
    pc, instrument = pair
    sim = make_sim(instrument, drop_after=1)
    thread = run_in_thread(sim)

    link = make_link(pc)
    link.trigger()
    link.wait_for_done()
    thread.join(timeout=2.0)

    with pytest.raises(InstrumentDisconnected):
        link.measure_point("x1.0 y1.0")
    pc.close()


# ── Jitter ───────────────────────────────────────────────────────────────────

def test_jitter_stays_within_bounds_and_never_goes_negative():
    left, right = socket.socketpair()
    try:
        sim = make_sim(right, acquire_s=0.1, jitter_s=0.5,
                       rng=random.Random(0))
        delays = [sim.next_acquire_s() for _ in range(200)]
    finally:
        left.close()
        right.close()
    assert all(0.0 <= d <= 0.6 for d in delays)
    assert len(set(delays)) > 1  # actually varying


def test_zero_jitter_is_exact():
    left, right = socket.socketpair()
    try:
        sim = make_sim(right, acquire_s=1.5)
        assert [sim.next_acquire_s() for _ in range(5)] == [1.5] * 5
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("kwargs", [{"acquire_s": -1.0}, {"jitter_s": -0.1}])
def test_negative_timings_are_rejected(kwargs):
    left, right = socket.socketpair()
    try:
        with pytest.raises(ValueError):
            SimulatedInstrument(right, **kwargs)
    finally:
        left.close()
        right.close()
