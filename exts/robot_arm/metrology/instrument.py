r"""
Client for the INT_Monitor measurement instrument.

Wire protocol (reverse-engineered from the original ``INT_Monitor/py``
scripts).  The PC opens a TCP *server* socket; the instrument monitor connects
to it as a client.  Afterwards the PC drives the exchange:

===========================  ==================================================
PC -> instrument             Meaning
===========================  ==================================================
``<axis command>\r\n``       Select which coordinate pair the PC will stream,
                             e.g. ``yx``, ``xz``, ``yz``.  Sent once, at
                             connection setup.
``x12.3 y-45.6\r\n``         Position label for the point about to be measured.
``1\r\n``                    Trigger: acquire one frame now.
===========================  ==================================================

===========================  ==================================================
instrument -> PC             Meaning
===========================  ==================================================
``done\r\n``                 Acquisition finished, PC may move on.
===========================  ==================================================

Fixes relative to the original scripts
--------------------------------------
* **Bounded waits.**  ``recv()`` was fully blocking with no timeout, so a
  missed ``done`` hung the scan forever.  :meth:`InstrumentLink.wait_for_done`
  takes a timeout and raises :class:`InstrumentTimeout`.
* **No lost bytes.**  The old code discarded everything received after the
  ``done`` token, then relied on a pre-trigger purge to clean up.  The leftover
  is now kept in an internal buffer and consumed by the next wait.
* **Triggers and responses stay in lockstep.**  The priming trigger sent at
  connection setup starts a real acquisition, but the old code only drained for
  50 ms before starting the scan.  Its ``done`` therefore arrived after the
  first point had been triggered and was consumed as that point's response, so
  every point reported the previous point's acquisition and the arm moved on
  while the instrument was still measuring.  :meth:`InstrumentLink.initialize`
  now waits for it.
* **Axis command matches the payload.**  ``Finall Rectangular Mesurment.py``
  announced ``yz`` but streamed ``x... y...`` labels, and
  ``Width X And Z Measurment .py`` announced the literal string
  ``output_line``.  The command is now derived from the :class:`AxisPair`
  actually being streamed.
* **Single-shot serving.**  The old ``while True: accept()`` loop silently
  re-ran a completed scan for every new client.  :func:`serve` accepts a fixed
  number of connections (one by default).
"""
from __future__ import annotations

import enum
import socket
import time
from typing import Callable, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "END_TOKEN",
    "TRIGGER_COMMAND",
    "AxisPair",
    "InstrumentError",
    "InstrumentDisconnected",
    "InstrumentTimeout",
    "InstrumentLink",
    "serve",
]

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6340
END_TOKEN = b"done\r\n"
TRIGGER_COMMAND = "1"
LINE_TERMINATOR = "\r\n"
RECV_CHUNK = 4096


class AxisPair(enum.Enum):
    """The two coordinates streamed to the instrument for each point.

    ``value`` is the axis command sent at connection setup.  The command tokens
    are the ones the original scripts used for each pair and are kept verbatim
    so the instrument-side configuration does not have to change.
    """

    XY = "yx"
    XZ = "xz"
    YZ = "yz"

    @property
    def command(self) -> str:
        """Axis-select command for this pair, without the line terminator."""
        return self.value

    @property
    def axes(self) -> Tuple[str, str]:
        """Axis letters in the order they appear in a point label."""
        return (self.name[0].lower(), self.name[1].lower())

    @property
    def indices(self) -> Tuple[int, int]:
        """Pose indices of :attr:`axes` (``x`` -> 0, ``y`` -> 1, ``z`` -> 2)."""
        first, second = self.axes
        return ("xyz".index(first), "xyz".index(second))

    def format_point(self, pose: Sequence[float], decimals: int = 1) -> str:
        """Render a pose as an instrument point label, e.g. ``"x12.3 y-45.6"``.

        ``pose`` may be a 3-value position or a full 6-value Cartesian pose.
        """
        if len(pose) < 3:
            raise ValueError(
                "Need at least 3 position values to build a point label, "
                f"got {list(pose)!r}"
            )
        return " ".join(
            f"{axis}{float(pose[index]):.{decimals}f}"
            for axis, index in zip(self.axes, self.indices)
        )

    @classmethod
    def parse(cls, text: str) -> "AxisPair":
        """Look up a pair by name (``"xy"``) or by axis command (``"yx"``)."""
        key = text.strip().lower()
        for member in cls:
            if key in (member.name.lower(), member.value):
                return member
        raise ValueError(
            f"Unknown axis pair {text!r}; expected one of "
            f"{[m.name.lower() for m in cls]}"
        )


class InstrumentError(RuntimeError):
    """Base class for INT_Monitor protocol failures."""


class InstrumentDisconnected(InstrumentError):
    """The instrument closed the connection mid-scan."""


class InstrumentTimeout(InstrumentError):
    """The instrument did not answer within the allotted time."""


class InstrumentLink:
    """Half-duplex request/response link to a connected INT_Monitor client.

    Parameters
    ----------
    conn:
        An already-accepted, connected stream socket.
    axis_pair:
        Which coordinate pair point labels carry.
    command_gap_s:
        Pause between the point label and the trigger pulse.  The instrument
        needs this to finish parsing the label; the original scripts used
        200 ms and that remains the default.
    response_timeout_s:
        How long to wait for ``done`` before giving up.
    """

    def __init__(
        self,
        conn: socket.socket,
        axis_pair: AxisPair = AxisPair.XY,
        *,
        command_gap_s: float = 0.2,
        response_timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.conn = conn
        self.axis_pair = axis_pair
        self.command_gap_s = command_gap_s
        self.response_timeout_s = response_timeout_s
        self._sleep = sleep
        self._clock = clock
        self._buffer = b""

    # -- low-level I/O -------------------------------------------------------

    def send_command(self, text: str) -> None:
        """Send one CRLF-terminated command."""
        try:
            self.conn.sendall((text + LINE_TERMINATOR).encode("utf-8"))
        except OSError as exc:
            raise InstrumentDisconnected(
                f"Failed to send {text!r} to the instrument: {exc}"
            ) from exc

    def drain(self, settle_s: float = 0.0) -> bytes:
        """Discard bytes the instrument sent before the next trigger.

        Returns whatever was discarded, for logging.  Never blocks.
        """
        if settle_s > 0:
            self._sleep(settle_s)
        discarded = self._buffer
        self._buffer = b""
        self.conn.setblocking(False)
        try:
            while True:
                try:
                    chunk = self.conn.recv(RECV_CHUNK)
                except (BlockingIOError, InterruptedError):
                    break
                except OSError:
                    break
                if not chunk:
                    break
                discarded += chunk
        finally:
            self.conn.setblocking(True)
        return discarded

    def wait_for_done(self, timeout_s: Optional[float] = None) -> float:
        """Block until ``done`` arrives; return the elapsed seconds.

        Raises :class:`InstrumentTimeout` if nothing arrives in time and
        :class:`InstrumentDisconnected` if the peer closes the connection.
        """
        timeout_s = self.response_timeout_s if timeout_s is None else timeout_s
        started = self._clock()
        deadline = started + timeout_s

        while True:
            index = self._buffer.find(END_TOKEN)
            if index >= 0:
                # Keep anything that arrived after the token for the next wait.
                self._buffer = self._buffer[index + len(END_TOKEN):]
                return self._clock() - started

            remaining = deadline - self._clock()
            if remaining <= 0:
                raise InstrumentTimeout(
                    f"No {END_TOKEN!r} from the instrument within "
                    f"{timeout_s:.1f} s (buffered {len(self._buffer)} bytes)"
                )

            self.conn.settimeout(remaining)
            try:
                chunk = self.conn.recv(RECV_CHUNK)
            except socket.timeout as exc:
                raise InstrumentTimeout(
                    f"No {END_TOKEN!r} from the instrument within "
                    f"{timeout_s:.1f} s"
                ) from exc
            except OSError as exc:
                raise InstrumentDisconnected(
                    f"Socket error while waiting for {END_TOKEN!r}: {exc}"
                ) from exc
            finally:
                self.conn.settimeout(None)

            if not chunk:
                raise InstrumentDisconnected(
                    "Instrument closed the connection while the scan was "
                    "waiting for an acquisition to finish"
                )
            self._buffer += chunk

    # -- protocol steps ------------------------------------------------------

    def initialize(
        self,
        settle_s: float = 0.5,
        priming_timeout_s: Optional[float] = None,
    ) -> bytes:
        """Select the axis pair, fire one priming trigger, await its ``done``.

        Returns any banner bytes the instrument sent back, which are consumed
        so they cannot be mistaken for a ``done`` later on.

        The priming trigger starts a real acquisition, so the instrument
        answers it with a ``done`` like any other.  The original scripts only
        drained for 50 ms before moving on, which is far shorter than an
        acquisition takes, so that ``done`` was still in flight when the first
        point was triggered and got consumed as *that* point's response.  Every
        subsequent point then reported the previous point's acquisition, the
        arm moved on while the instrument was still measuring, and the final
        point's ``done`` was never awaited at all.  Waiting for it here keeps
        one trigger matched to one ``done`` for the rest of the scan.

        ``priming_timeout_s`` bounds that wait and defaults to
        :attr:`response_timeout_s`.  Pass ``0`` to skip it, for an instrument
        that does not answer the priming trigger; the link then falls back to
        the old drain-only behaviour.
        """
        self.send_command(self.axis_pair.command)
        self._sleep(settle_s)
        banner = self.drain()
        self.send_command(TRIGGER_COMMAND)

        timeout_s = (
            self.response_timeout_s if priming_timeout_s is None
            else priming_timeout_s
        )
        if timeout_s > 0:
            try:
                self.wait_for_done(timeout_s)
            except InstrumentTimeout:
                print(
                    f"[instrument] No {END_TOKEN!r} for the priming trigger "
                    f"within {timeout_s:.1f} s; continuing without it."
                )
        return banner + self.drain(settle_s=0.05)

    def trigger(self) -> None:
        """Send the acquisition trigger pulse."""
        self.send_command(TRIGGER_COMMAND)

    def send_point(self, label: str) -> None:
        """Send a point label, pause, then trigger the acquisition."""
        self.send_command(label)
        self._sleep(self.command_gap_s)
        self.trigger()

    def measure_point(
        self,
        label: str,
        timeout_s: Optional[float] = None,
    ) -> float:
        """Full single-point handshake; returns the acquisition time in seconds.

        Stale bytes are purged first so a late ``done`` from the previous point
        can never be mistaken for this point's response.
        """
        self.drain()
        self.send_point(label)
        return self.wait_for_done(timeout_s)

    def close(self) -> None:
        """Close the underlying socket, ignoring an already-closed socket."""
        try:
            self.conn.close()
        except OSError:
            pass

    def __enter__(self) -> "InstrumentLink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def serve(
    handler: Callable[[socket.socket, object], None],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    max_connections: int = 1,
    accept_timeout_s: Optional[float] = None,
    backlog: int = 5,
) -> int:
    """Accept instrument connections and hand each one to ``handler``.

    Returns the number of connections served.  Unlike the original
    ``while True`` loops this stops after ``max_connections`` (one by default),
    so a finished scan is never silently repeated.  Pass ``max_connections=0``
    for the old unbounded behaviour.

    ``accept_timeout_s`` bounds the wait for a client; ``None`` waits forever.
    """
    served = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(backlog)
        print(f"[instrument] TCP server listening on {host}:{port}")

        while max_connections == 0 or served < max_connections:
            if accept_timeout_s is not None:
                server.settimeout(accept_timeout_s)
            try:
                conn, addr = server.accept()
            except socket.timeout as exc:
                raise InstrumentTimeout(
                    f"No instrument connected within {accept_timeout_s:.1f} s"
                ) from exc
            finally:
                server.settimeout(None)

            print(f"[instrument] Connection from {addr}")
            with conn:
                conn.settimeout(None)
                handler(conn, addr)
            served += 1

    return served
