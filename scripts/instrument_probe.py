r"""
Diagnose the INT_Monitor TCP link, with no robot and no Isaac Sim involved.

The scanners in ``scripts/`` connect and home the robot *before* they open the
TCP server, so a robot problem looks exactly like a network problem: the
instrument never gets a connection and no data reaches Python.  This probe
opens the same server on the same port, by itself, and prints every byte that
crosses the link with timestamps.

Typical use
-----------
Start the probe first, then tell INT_Monitor to connect::

    micromamba run -n RobotArm python scripts/instrument_probe.py --listen-only
    micromamba run -n RobotArm python scripts/instrument_probe.py --handshake --points 3

``--listen-only``  accept the connection and dump whatever arrives; send nothing.
``--handshake``    send the axis command + priming trigger, then wait for ``done``.
``--points N``     after the handshake, run N fake measurement points (no motion),
                   which is the exact per-point exchange ``run_scan`` performs.

Exit status is 0 only if every expected ``done`` arrived.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import `metrology` as a top-level package: exts/robot_arm/__init__.py pulls
# in Isaac Sim, which this probe deliberately does not need.
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "robot_arm"))

from metrology import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    END_TOKEN,
    AxisPair,
    InstrumentDisconnected,
    InstrumentLink,
    InstrumentTimeout,
)

T0 = time.monotonic()


def stamp() -> str:
    return f"[{time.monotonic() - T0:7.3f}s]"


def show(direction: str, payload: bytes) -> None:
    text = payload.decode("utf-8", errors="replace")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    print(f"{stamp()} {direction} {len(payload):4d} B  {text!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the INT_Monitor TCP link (no robot needed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Bind address; use 0.0.0.0 to accept any interface")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    parser.add_argument("--axis-pair", default="xy",
                        help="Axis pair whose command is sent at setup (xy, xz, yz)")
    parser.add_argument("--accept-timeout", type=float, default=120.0,
                        help="Seconds to wait for INT_Monitor to connect")
    parser.add_argument("--response-timeout", type=float, default=30.0,
                        help="Seconds to wait for each 'done'")
    parser.add_argument("--listen-only", action="store_true",
                        help="Never transmit; just dump what the instrument sends")
    parser.add_argument("--handshake", action="store_true",
                        help="Send the axis command + priming trigger and await 'done'")
    parser.add_argument("--points", type=int, default=0,
                        help="Fake measurement points to run after the handshake")
    return parser


def listen_only(conn: socket.socket, seconds: float) -> int:
    """Dump everything the instrument sends, unprompted, for `seconds`."""
    print(f"{stamp()} Listening only - nothing will be transmitted.")
    deadline = time.monotonic() + seconds
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        conn.settimeout(remaining)
        try:
            chunk = conn.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            print(f"{stamp()} Instrument closed the connection.")
            break
        show("RX", chunk)
        total += len(chunk)
    print(f"{stamp()} {total} byte(s) received.")
    if total == 0:
        print("  INT_Monitor connected but sent nothing on its own.  That is "
              "expected: it only answers a trigger.  Re-run with --handshake.")
    return 0


def run_probe(args: argparse.Namespace, conn: socket.socket) -> int:
    pair = AxisPair.parse(args.axis_pair)

    if args.listen_only:
        return listen_only(conn, args.accept_timeout)

    link = InstrumentLink(conn, pair, response_timeout_s=args.response_timeout)
    failures = 0

    if args.handshake or args.points:
        print(f"{stamp()} Sending axis command {pair.command!r}, then trigger '1'")
        show("TX", (pair.command + "\r\n").encode())
        show("TX", b"1\r\n")
        banner = link.initialize()
        if banner:
            show("RX", banner)
            print(f"{stamp()} (banner consumed by the priming drain)")
        else:
            print(f"{stamp()} No banner within the 50 ms priming drain "
                  f"(normal; the priming 'done' arrives later)")

    for index in range(args.points):
        label = pair.format_point([index * 1.0, index * 2.0, 0.0])
        print(f"{stamp()} --- point {index + 1}/{args.points} ---")
        show("TX", (label + "\r\n").encode())
        show("TX", b"1\r\n")
        try:
            elapsed = link.measure_point(label)
        except InstrumentTimeout as exc:
            print(f"{stamp()} TIMEOUT  {exc}")
            failures += 1
            break
        except InstrumentDisconnected as exc:
            print(f"{stamp()} DISCONNECTED  {exc}")
            failures += 1
            break
        show("RX", END_TOKEN)
        print(f"{stamp()} done after {elapsed:.3f} s")

    if args.handshake and not args.points:
        try:
            elapsed = link.wait_for_done()
        except InstrumentTimeout as exc:
            print(f"{stamp()} TIMEOUT  {exc}")
            failures += 1
        except InstrumentDisconnected as exc:
            print(f"{stamp()} DISCONNECTED  {exc}")
            failures += 1
        else:
            show("RX", END_TOKEN)
            print(f"{stamp()} Priming acquisition answered in {elapsed:.3f} s")

    return 1 if failures else 0


def open_listeners(host: str, port: int) -> list:
    """Listening sockets for `host`, covering IPv4 and IPv6 separately.

    ``localhost`` resolves to ``::1`` before ``127.0.0.1`` on Windows, so an
    IPv4-only server (what the original scripts opened) silently misses a client
    that dials the IPv6 loopback.  Both families are bound where possible, each
    as a socket of its own with ``IPV6_V6ONLY`` set, so neither bind can steal
    the other's port.
    """
    wildcards = {"", "0.0.0.0", "::", "any", "all"}
    if host.lower() in wildcards:
        candidates = [(socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")]
    elif host.lower() == "localhost":
        candidates = [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
    else:
        try:
            family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][0]
        except OSError:
            family = socket.AF_INET
        candidates = [(family, host)]

    listeners = []
    for family, address in candidates:
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            # Keep the two sockets independent rather than dual-stacking one.
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            sock.bind((address, port))
            sock.listen(5)
        except OSError as exc:
            sock.close()
            print(f"{stamp()} Could not bind {address}:{port} - {exc}")
            continue
        listeners.append(sock)
        print(f"{stamp()} Listening on {address}:{port}")
    return listeners


def main(argv=None) -> int:
    import select

    args = build_parser().parse_args(argv)

    listeners = open_listeners(args.host, args.port)
    if not listeners:
        print(f"[ERROR] Could not listen on {args.host}:{args.port}.",
              file=sys.stderr)
        print("        Another scan script may still be running, or the "
              "address is not one of this machine's.", file=sys.stderr)
        return 1

    try:
        print(f"{stamp()} Waiting up to {args.accept_timeout:.0f} s for INT_Monitor")
        print("          Now start the measurement in INT_Monitor so it connects.")
        ready, _, _ = select.select(listeners, [], [], args.accept_timeout)
        if not ready:
            print(f"\n{stamp()} No connection within {args.accept_timeout:.0f} s.",
                  file=sys.stderr)
            print("  The instrument never reached this server.  Check, in order:",
                  file=sys.stderr)
            print("    1. INT_Monitor's TCP settings: Active? on, and Address/Port "
                  f"pointing at {args.port}", file=sys.stderr)
            print("    2. It connects only when a measurement is actually started, "
                  "not when Active? is switched on", file=sys.stderr)
            print("    3. Toggle Active? off and on again with this probe already "
                  "listening - a client that failed to connect earlier may not "
                  "retry on its own", file=sys.stderr)
            print("    4. If it targets a LAN address rather than loopback, "
                  "re-run with --host 0.0.0.0", file=sys.stderr)
            return 1

        conn, addr = ready[0].accept()
        family = "IPv6" if ready[0].family == socket.AF_INET6 else "IPv4"
        print(f"{stamp()} Connection from {addr} over {family}")
        with conn:
            conn.settimeout(None)
            return run_probe(args, conn)
    finally:
        for sock in listeners:
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
