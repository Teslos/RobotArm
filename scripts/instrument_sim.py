#!/usr/bin/env python3
r"""
Stand-in for the INT_Monitor program: answers every trigger after a fixed
acquisition time, so a scan can be rehearsed without the instrument.

This is the mirror image of ``scripts/instrument_probe.py``.  The probe plays
the *PC* side (it listens, like the scan scripts do); this plays the
*instrument* side (it dials in, like INT_Monitor does).  Run it next to a real
scan and the arm moves for real while the measurement is simulated::

    # terminal 1 - the real scan, real arm
    micromamba run -n RobotArm python scripts/scan_raster.py --n-fast 10 --n-slow 10

    # terminal 2 - stands in for INT_Monitor, 2 s per acquisition
    micromamba run -n RobotArm python scripts/instrument_sim.py --acquire-time 2.0

The scan scripts bind their socket only *after* the arm has homed, so the
simulator retries the connection for ``--retry-for`` seconds rather than giving
up on the first refusal.

What it models
--------------
* the axis-select command and the banner an instrument may send back,
* the priming trigger ``initialize()`` fires, answered like any other,
* one ``done\r\n`` per trigger, after ``--acquire-time`` seconds,
* optional jitter, so consecutive points do not all take exactly the same time.

It can also misbehave on purpose, to exercise the scanner's error paths:

===========================  ==================================================
``--silent-after N``         Stop answering after N triggers -> the scan should
                             raise InstrumentTimeout at --response-timeout.
``--drop-after N``           Hang up after answering N triggers -> the scan
                             should raise InstrumentDisconnected.
===========================  ==================================================

Neither fault is the default; without them the simulator answers forever.
"""
from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `metrology` is imported as a top-level package rather than as
# `exts.robot_arm.metrology`, because exts/robot_arm/__init__.py pulls in
# Isaac Sim.  Simulating the instrument does not need a simulator installed.
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "robot_arm"))

from metrology.instrument import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    END_TOKEN,
    TRIGGER_COMMAND,
    AxisPair,
)

RECV_CHUNK = 4096
AXIS_COMMANDS = {pair.command for pair in AxisPair}


def stamp(started: float = 0.0) -> str:
    """``[  12.345s]`` since the simulator started, to line up with the scan."""
    return f"[{time.monotonic() - started:7.3f}s]"


class SimulatedInstrument:
    """Answers the INT_Monitor half of the handshake on an open socket.

    The socket is supplied already connected, so tests can drive this over a
    ``socket.socketpair()`` with an instant ``sleep``.
    """

    def __init__(
        self,
        conn: socket.socket,
        *,
        acquire_s: float = 2.0,
        jitter_s: float = 0.0,
        max_points: int = 0,
        silent_after: int = 0,
        drop_after: int = 0,
        banner: str = "",
        sleep=time.sleep,
        log=print,
        rng=None,
    ) -> None:
        if acquire_s < 0.0:
            raise ValueError(f"acquire_s must be >= 0, got {acquire_s}")
        if jitter_s < 0.0:
            raise ValueError(f"jitter_s must be >= 0, got {jitter_s}")
        self.conn = conn
        self.acquire_s = acquire_s
        self.jitter_s = jitter_s
        self.max_points = max_points
        self.silent_after = silent_after
        self.drop_after = drop_after
        self.banner = banner
        self._sleep = sleep
        self._log = log
        self._rng = rng or random.Random()
        self._buffer = b""
        self.triggers = 0
        self.answered = 0
        self.label = ""

    # -- helpers -------------------------------------------------------------

    def next_acquire_s(self) -> float:
        """The acquisition time for the next point, jitter included."""
        if not self.jitter_s:
            return self.acquire_s
        return max(0.0, self.acquire_s + self._rng.uniform(-self.jitter_s, self.jitter_s))

    def _read_lines(self):
        """Yield complete CRLF- or LF-terminated commands; b'' when closed."""
        chunk = self.conn.recv(RECV_CHUNK)
        if not chunk:
            return None
        self._buffer += chunk
        lines = []
        while True:
            index = self._buffer.find(b"\n")
            if index < 0:
                break
            line, self._buffer = self._buffer[:index], self._buffer[index + 1:]
            lines.append(line.strip().decode("utf-8", errors="replace"))
        return [line for line in lines if line]

    # -- the loop ------------------------------------------------------------

    def handle_trigger(self) -> bool:
        """Acquire one point.  Returns False when the simulator should stop."""
        self.triggers += 1
        what = self.label or "priming trigger"

        if self.silent_after and self.triggers > self.silent_after:
            self._log(f"  trigger {self.triggers}: {what} -> staying silent "
                      f"(--silent-after {self.silent_after})")
            return True

        delay = self.next_acquire_s()
        self._sleep(delay)
        try:
            self.conn.sendall(END_TOKEN)
        except OSError as exc:
            self._log(f"  send failed, the scan hung up: {exc}")
            return False
        self.answered += 1
        self._log(f"  trigger {self.triggers}: {what} -> done after {delay:.2f} s")

        if self.drop_after and self.answered >= self.drop_after:
            self._log(f"  hanging up after {self.answered} point(s) "
                      f"(--drop-after {self.drop_after})")
            # Close here rather than leaving it to the caller: the modelled
            # fault *is* the hang-up, so the scan must see the socket go away.
            self.conn.close()
            return False
        if self.max_points and self.answered >= self.max_points:
            self._log(f"  reached --points {self.max_points}, stopping")
            return False
        return True

    def run(self) -> int:
        """Serve until the scan disconnects or a stop condition trips.

        Returns the number of acquisitions answered.
        """
        if self.banner:
            self.conn.sendall((self.banner + "\r\n").encode("utf-8"))
            self._log(f"  banner sent: {self.banner!r}")

        while True:
            try:
                lines = self._read_lines()
            except OSError as exc:
                self._log(f"  connection error: {exc}")
                return self.answered
            if lines is None:
                self._log("  the scan closed the connection")
                return self.answered

            for line in lines:
                if line == TRIGGER_COMMAND:
                    if not self.handle_trigger():
                        return self.answered
                elif line in AXIS_COMMANDS:
                    self._log(f"  axis pair selected: {line!r}")
                else:
                    self.label = line


def connect_with_retry(host: str, port: int, retry_for_s: float, log=print) -> socket.socket:
    """Dial the scan script, waiting for it to finish homing and bind."""
    deadline = time.monotonic() + retry_for_s
    attempt = 0
    while True:
        attempt += 1
        try:
            return socket.create_connection((host, port), timeout=5.0)
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise OSError(
                    f"Could not reach {host}:{port} within {retry_for_s:.0f} s "
                    f"({exc}). Is the scan script running? It binds the socket "
                    f"only after the arm has homed."
                ) from exc
            if attempt == 1:
                log(f"  {host}:{port} not up yet, retrying for "
                    f"{retry_for_s:.0f} s ...")
            time.sleep(1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate INT_Monitor with a fixed acquisition time",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--acquire-time", type=float, default=2.0, metavar="SECONDS",
                        help="Seconds to 'measure' before answering each trigger")
    parser.add_argument("--jitter", type=float, default=0.0, metavar="SECONDS",
                        help="Vary each acquisition by +/- this much")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Address the scan script is listening on")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port the scan script is listening on")
    parser.add_argument("--retry-for", type=float, default=120.0, metavar="SECONDS",
                        help="Keep retrying the connection for this long; the "
                             "scan binds only after the arm has homed")
    parser.add_argument("--points", type=int, default=0, metavar="N",
                        help="Stop after N acquisitions (0 = until the scan ends)")
    parser.add_argument("--banner", default="",
                        help="Line to send on connect, as a chatty instrument would")

    faults = parser.add_argument_group("fault injection")
    faults.add_argument("--silent-after", type=int, default=0, metavar="N",
                        help="Stop answering after N triggers, to exercise the "
                             "scan's --response-timeout")
    faults.add_argument("--drop-after", type=int, default=0, metavar="N",
                        help="Hang up after N acquisitions, to exercise the "
                             "scan's disconnect handling")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.acquire_time < 0.0:
        print("[ERROR] --acquire-time cannot be negative.", file=sys.stderr)
        return 2
    if args.jitter < 0.0:
        print("[ERROR] --jitter cannot be negative.", file=sys.stderr)
        return 2

    started = time.monotonic()

    def log(message: str) -> None:
        print(f"{stamp(started)} {message}", flush=True)

    print("=" * 72)
    print("INT_Monitor SIMULATOR - no instrument is involved")
    print(f"  Target   : {args.host}:{args.port}")
    print(f"  Acquire  : {args.acquire_time:.2f} s"
          + (f" +/- {args.jitter:.2f} s" if args.jitter else ""))
    print("=" * 72)

    try:
        conn = connect_with_retry(args.host, args.port, args.retry_for, log=log)
    except OSError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    log(f"  connected to {args.host}:{args.port}")
    sim = SimulatedInstrument(
        conn,
        acquire_s=args.acquire_time,
        jitter_s=args.jitter,
        max_points=args.points,
        silent_after=args.silent_after,
        drop_after=args.drop_after,
        banner=args.banner,
        log=log,
    )
    try:
        conn.settimeout(None)
        answered = sim.run()
    except KeyboardInterrupt:
        log("  interrupted")
        answered = sim.answered
    finally:
        conn.close()

    log(f"  {answered} acquisition(s) answered, "
        f"{sim.triggers} trigger(s) seen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
