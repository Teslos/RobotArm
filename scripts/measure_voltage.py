r"""
Measure voltages with an NI USB X Series DAQ (no robot, no Isaac Sim).

Needs the NI-DAQmx driver (ni.com) and ``pip install nidaqmx``.

Typical use
-----------
    micromamba run -n RobotArm python scripts/measure_voltage.py --list
    micromamba run -n RobotArm python scripts/measure_voltage.py --channels Dev1/ai0
    micromamba run -n RobotArm python scripts/measure_voltage.py --channels Dev1/ai0:1 `
        --terminal diff --samples 200 --repeat 10 --output results/voltage.csv

Each row is the mean of ``--samples`` clocked samples per channel.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import `metrology` as a top-level package: exts/robot_arm/__init__.py pulls
# in Isaac Sim, which this script deliberately does not need.
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "robot_arm"))

from metrology.daq import (  # noqa: E402
    TERMINAL_CONFIGS,
    DaqError,
    DaqSettings,
    VoltageReader,
    list_devices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure voltages with an NI USB X Series DAQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true",
                        help="List the DAQ devices the driver sees, then exit")
    parser.add_argument("--channels", nargs="+", default=["Dev1/ai0"],
                        help="Physical channels, e.g. Dev1/ai0 or Dev1/ai0:3")
    parser.add_argument("--terminal", choices=TERMINAL_CONFIGS, default="default",
                        help="Input wiring")
    parser.add_argument("--min-v", type=float, default=-10.0, help="Range minimum, V")
    parser.add_argument("--max-v", type=float, default=10.0, help="Range maximum, V")
    parser.add_argument("--rate", type=float, default=1000.0, metavar="HZ",
                        help="Sample clock rate")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples averaged per reading")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of readings")
    parser.add_argument("--interval", type=float, default=0.0, metavar="SECONDS",
                        help="Pause between readings")
    parser.add_argument("--output", default=None, metavar="CSV",
                        help="Also write the readings to this CSV file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.list:
            devices = list_devices()
            print("\n".join(devices) if devices else "No DAQ devices found.")
            return 0

        settings = DaqSettings(
            min_v=args.min_v, max_v=args.max_v, terminal=args.terminal,
            sample_rate_hz=args.rate,
        )
        rows = []
        with VoltageReader(args.channels, settings) as daq:
            n = daq.n_channels
            print(f"[daq] {n} channel(s), {args.samples} samples @ {args.rate:g} Hz")
            started = time.monotonic()
            for index in range(args.repeat):
                volts = daq.read_mean(args.samples)
                elapsed = time.monotonic() - started
                rows.append([index, f"{elapsed:.3f}", *[f"{v:.6f}" for v in volts]])
                print(f"[daq] {index + 1:>4}/{args.repeat}  "
                      + "  ".join(f"{v:+.5f} V" for v in volts))
                if args.interval > 0 and index < args.repeat - 1:
                    time.sleep(args.interval)

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "time_s", *[f"ch{i}_V" for i in range(n)]])
                writer.writerows(rows)
            print(f"[daq] wrote {len(rows)} rows to {args.output}")
    except DaqError as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
