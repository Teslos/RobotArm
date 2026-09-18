"""
Unit tests for the INT_Monitor protocol client.

The socket-level tests use a real ``socket.socketpair()`` so timeouts,
non-blocking drains and partial reads behave exactly as they do on the wire.
"""
from __future__ import annotations

import socket
import threading

import pytest

from exts.robot_arm.metrology.instrument import (
    END_TOKEN,
    TRIGGER_COMMAND,
    AxisPair,
    InstrumentDisconnected,
    InstrumentLink,
    InstrumentTimeout,
    serve,
)


@pytest.fixture
def pair():
    """A connected (pc_side, instrument_side) socket pair."""
    left, right = socket.socketpair()
    try:
        yield left, right
    finally:
        left.close()
        right.close()


def make_link(conn, **kwargs):
    """A link whose sleeps are instant, so tests do not wait on pacing gaps."""
    kwargs.setdefault("command_gap_s", 0.0)
    kwargs.setdefault("response_timeout_s", 2.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return InstrumentLink(conn, **kwargs)


# ── AxisPair ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair_name,command,axes,indices", [
    ("XY", "yx", ("x", "y"), (0, 1)),
    ("XZ", "xz", ("x", "z"), (0, 2)),
    ("YZ", "yz", ("y", "z"), (1, 2)),
])
def test_axis_pair_metadata(pair_name, command, axes, indices):
    member = AxisPair[pair_name]
    assert member.command == command
    assert member.axes == axes
    assert member.indices == indices


def test_format_point_uses_the_right_pose_indices():
    pose = [1.25, -2.5, 3.75, 0.0, 0.0, 0.0]
    assert AxisPair.XY.format_point(pose) == "x1.2 y-2.5"
    assert AxisPair.XZ.format_point(pose) == "x1.2 z3.8"
    assert AxisPair.YZ.format_point(pose) == "y-2.5 z3.8"


def test_format_point_honours_decimals():
    assert AxisPair.XY.format_point([1.23456, 2.0, 3.0], decimals=3) == "x1.235 y2.000"


def test_format_point_accepts_a_bare_position():
    assert AxisPair.YZ.format_point([1.0, 2.0, 3.0]) == "y2.0 z3.0"


def test_format_point_rejects_short_input():
    with pytest.raises(ValueError, match="at least 3 position values"):
        AxisPair.XY.format_point([1.0, 2.0])


@pytest.mark.parametrize("text", ["xy", "XY", " yx ", "yz", "xz"])
def test_axis_pair_parse(text):
    assert isinstance(AxisPair.parse(text), AxisPair)


def test_axis_pair_parse_rejects_the_old_bogus_command():
    """Width X And Z Measurment .py sent the literal string 'output_line'."""
    with pytest.raises(ValueError, match="Unknown axis pair"):
        AxisPair.parse("output_line")


# ── Sending ──────────────────────────────────────────────────────────────────

def test_send_point_emits_label_then_trigger(pair):
    pc, instrument = pair
    make_link(pc).send_point("x1.0 y2.0")
    assert instrument.recv(1024) == b"x1.0 y2.0\r\n1\r\n"


def test_initialize_announces_the_axis_pair_matching_the_payload(pair):
    """Finall Rectangular Mesurment.py announced 'yz' while streaming 'x.. y..'."""
    pc, instrument = pair
    link = make_link(pc, axis_pair=AxisPair.XY)
    link.initialize(settle_s=0.0)
    sent = instrument.recv(1024).decode()
    assert sent == f"{AxisPair.XY.command}\r\n{TRIGGER_COMMAND}\r\n"


def test_initialize_consumes_the_instrument_banner(pair):
    pc, instrument = pair
    instrument.sendall(b"READY\r\n")
    link = make_link(pc)
    assert b"READY" in link.initialize(settle_s=0.0)

    # The banner must not be mistaken for a later 'done'.
    instrument.sendall(END_TOKEN)
    assert link.wait_for_done() >= 0.0


def test_send_command_on_a_closed_socket_raises_disconnected(pair):
    pc, _instrument = pair
    link = make_link(pc)
    pc.close()
    with pytest.raises(InstrumentDisconnected):
        link.send_command("x1.0 y2.0")


# ── Waiting for 'done' ───────────────────────────────────────────────────────

def test_wait_for_done_returns_elapsed(pair):
    pc, instrument = pair
    instrument.sendall(END_TOKEN)
    assert make_link(pc).wait_for_done() >= 0.0


def test_wait_for_done_reassembles_a_split_token(pair):
    pc, instrument = pair
    link = make_link(pc)
    instrument.sendall(b"do")
    instrument.sendall(b"ne\r\n")
    assert link.wait_for_done() >= 0.0


def test_wait_for_done_keeps_bytes_that_follow_the_token(pair):
    """The originals discarded the tail; two 'done's in one packet were lost."""
    pc, instrument = pair
    link = make_link(pc)
    instrument.sendall(END_TOKEN + END_TOKEN)
    link.wait_for_done()
    # The second token is already buffered: this must not block or time out.
    assert link.wait_for_done(timeout_s=0.5) >= 0.0


def test_wait_for_done_times_out_instead_of_hanging(pair):
    """The original scripts blocked in recv() forever if 'done' never came."""
    pc, _instrument = pair
    link = make_link(pc, response_timeout_s=0.15)
    with pytest.raises(InstrumentTimeout, match="within 0.1 s"):
        link.wait_for_done()


def test_wait_for_done_timeout_is_overridable(pair):
    pc, _instrument = pair
    link = make_link(pc, response_timeout_s=60.0)
    with pytest.raises(InstrumentTimeout):
        link.wait_for_done(timeout_s=0.1)


def test_wait_for_done_detects_a_closed_peer(pair):
    pc, instrument = pair
    link = make_link(pc)
    instrument.shutdown(socket.SHUT_WR)
    with pytest.raises(InstrumentDisconnected, match="closed the connection"):
        link.wait_for_done()


def test_socket_is_left_blocking_after_a_timeout(pair):
    """A leaked timeout would make the next point fail for the wrong reason."""
    pc, instrument = pair
    link = make_link(pc, response_timeout_s=0.1)
    with pytest.raises(InstrumentTimeout):
        link.wait_for_done()
    assert pc.gettimeout() is None
    instrument.sendall(END_TOKEN)
    assert link.wait_for_done(timeout_s=2.0) >= 0.0


# ── Draining ─────────────────────────────────────────────────────────────────

def test_drain_removes_stale_bytes_and_does_not_block(pair):
    pc, instrument = pair
    link = make_link(pc)
    instrument.sendall(b"stale garbage")
    assert b"stale garbage" in link.drain()
    assert link.drain() == b""
    assert pc.gettimeout() is None


def test_drain_discards_a_stale_done_so_it_cannot_ack_the_next_point(pair):
    pc, instrument = pair
    link = make_link(pc, response_timeout_s=0.15)
    instrument.sendall(END_TOKEN)      # late reply to the *previous* point
    link.drain()
    with pytest.raises(InstrumentTimeout):
        link.wait_for_done()


# ── Full handshake ───────────────────────────────────────────────────────────

def test_measure_point_round_trip(pair):
    pc, instrument = pair
    link = make_link(pc)

    def responder():
        instrument.recv(1024)
        instrument.sendall(END_TOKEN)

    thread = threading.Thread(target=responder)
    thread.start()
    try:
        elapsed = link.measure_point("x1.0 y2.0")
    finally:
        thread.join(timeout=5.0)
    assert elapsed >= 0.0


# ── serve ────────────────────────────────────────────────────────────────────

def _client(port: int) -> None:
    with socket.create_connection(("localhost", port), timeout=5.0) as sock:
        sock.sendall(END_TOKEN)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


def test_serve_stops_after_one_connection_by_default():
    """The originals looped on accept() forever and silently re-ran the scan."""
    port = _free_port()
    seen = []

    def handler(conn, addr):
        seen.append(addr)

    thread = threading.Thread(target=lambda: serve(handler, "localhost", port))
    thread.start()
    try:
        _client(port)
    finally:
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(seen) == 1


def test_serve_accept_timeout_raises():
    port = _free_port()
    with pytest.raises(InstrumentTimeout, match="No instrument connected"):
        serve(lambda c, a: None, "localhost", port, accept_timeout_s=0.2)
