"""
Voltage measurement with a National Instruments USB X Series DAQ.

A thin wrapper over the official ``nidaqmx`` package (which itself needs the
NI-DAQmx driver installed on Windows).  ``nidaqmx`` is imported lazily, so this
module -- and the tests -- load on machines without the driver.

Typical use::

    with VoltageReader("Dev1/ai0") as daq:
        volts = daq.read_voltage()              # one value
        mean = daq.read_mean(n_samples=100)     # averaged burst
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "DaqError",
    "DaqSettings",
    "VoltageReader",
    "add_daq_arguments",
    "list_devices",
    "open_reader_from_args",
]

TERMINAL_CONFIGS = ("default", "rse", "nrse", "diff", "pseudo_diff")


class DaqError(RuntimeError):
    """Raised for DAQ setup problems (missing driver, bad arguments)."""


@dataclass
class DaqSettings:
    """Acquisition knobs for :class:`VoltageReader`."""

    min_v: float = -10.0
    max_v: float = 10.0
    """Expected signal range; USB X Series inputs support up to ±10 V."""

    terminal: str = "default"
    """One of :data:`TERMINAL_CONFIGS`; ``diff`` is the least noisy."""

    sample_rate_hz: float = 1000.0
    """Sample clock for burst reads (:meth:`VoltageReader.read_samples`)."""

    timeout_s: float = 10.0
    """Per-read timeout."""

    def __post_init__(self) -> None:
        if self.min_v >= self.max_v:
            raise DaqError(
                f"min_v must be below max_v, got [{self.min_v}, {self.max_v}]"
            )
        if self.terminal not in TERMINAL_CONFIGS:
            raise DaqError(
                f"terminal must be one of {TERMINAL_CONFIGS}, got {self.terminal!r}"
            )
        if self.sample_rate_hz <= 0:
            raise DaqError("sample_rate_hz must be positive")


def _load_nidaqmx():
    try:
        import nidaqmx
        import nidaqmx.constants
        import nidaqmx.system
    except ImportError as exc:  # pragma: no cover - depends on the PC
        raise DaqError(
            "The 'nidaqmx' package is missing. Run `pip install nidaqmx` and "
            "install the NI-DAQmx driver from ni.com."
        ) from exc
    return nidaqmx


def list_devices() -> List[str]:
    """Names of the DAQ devices the driver can see, e.g. ``['Dev1']``."""
    nidaqmx = _load_nidaqmx()
    return [device.name for device in nidaqmx.system.System.local().devices]


def add_daq_arguments(parser) -> None:
    """Add the shared ``--daq-*`` options to an ``argparse`` parser."""
    group = parser.add_argument_group("voltage measurement (NI DAQ)")
    group.add_argument("--daq-channels", nargs="+", metavar="CH",
                       help="Read these analog inputs at every point, e.g. "
                            "Dev1/ai0 or Dev1/ai0:1; omit to scan without a DAQ")
    group.add_argument("--daq-terminal", choices=TERMINAL_CONFIGS,
                       default="default", help="Input wiring")
    group.add_argument("--daq-range", type=float, nargs=2, default=(-10.0, 10.0),
                       metavar=("MIN_V", "MAX_V"), help="Expected signal range")
    group.add_argument("--daq-rate", type=float, default=1000.0, metavar="HZ",
                       help="Sample clock rate")
    group.add_argument("--daq-samples", type=int, default=100,
                       help="Samples averaged per point")


def open_reader_from_args(args) -> Optional["VoltageReader"]:
    """Open the DAQ named by the ``--daq-*`` options, or ``None`` if unused."""
    if not args.daq_channels:
        return None
    settings = DaqSettings(
        min_v=args.daq_range[0], max_v=args.daq_range[1],
        terminal=args.daq_terminal, sample_rate_hz=args.daq_rate,
    )
    return VoltageReader(args.daq_channels, settings).open()


class VoltageReader:
    """Analog-input voltage reader for one or more channels.

    Parameters
    ----------
    channels:
        A physical channel or a list, e.g. ``"Dev1/ai0"``, ``"Dev1/ai0:3"`` or
        ``["Dev1/ai0", "Dev1/ai2"]``.
    settings:
        Range / wiring / sample-rate options.
    """

    def __init__(
        self,
        channels: Union[str, Sequence[str]],
        settings: Optional[DaqSettings] = None,
    ) -> None:
        self.settings = settings or DaqSettings()
        self.channels = [channels] if isinstance(channels, str) else list(channels)
        if not self.channels:
            raise DaqError("At least one channel is required")
        self._nidaqmx = _load_nidaqmx()
        self._task = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "VoltageReader":
        """Create the task.  Idempotent."""
        if self._task is not None:
            return self
        constants = self._nidaqmx.constants
        terminal = {
            "default": constants.TerminalConfiguration.DEFAULT,
            "rse": constants.TerminalConfiguration.RSE,
            "nrse": constants.TerminalConfiguration.NRSE,
            "diff": constants.TerminalConfiguration.DIFF,
            "pseudo_diff": constants.TerminalConfiguration.PSEUDO_DIFF,
        }[self.settings.terminal]

        task = self._nidaqmx.Task()
        try:
            for channel in self.channels:
                task.ai_channels.add_ai_voltage_chan(
                    channel,
                    terminal_config=terminal,
                    min_val=self.settings.min_v,
                    max_val=self.settings.max_v,
                )
        except Exception:
            task.close()
            raise
        self._task = task
        return self

    def close(self) -> None:
        if self._task is not None:
            self._task.close()
            self._task = None

    def __enter__(self) -> "VoltageReader":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- reads --------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        self._require_open()
        return len(self._task.ai_channels)

    def read_voltage(self) -> np.ndarray:
        """One on-demand reading per channel, shape ``(n_channels,)`` in volts."""
        self._require_open()
        value = self._task.read(timeout=self.settings.timeout_s)
        return np.atleast_1d(np.asarray(value, dtype=float))

    def read_samples(self, n_samples: int) -> np.ndarray:
        """A clocked burst, shape ``(n_channels, n_samples)`` in volts."""
        self._require_open()
        if n_samples < 1:
            raise DaqError("n_samples must be at least 1")
        constants = self._nidaqmx.constants
        self._task.timing.cfg_samp_clk_timing(
            self.settings.sample_rate_hz,
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )
        data = self._task.read(
            number_of_samples_per_channel=n_samples,
            timeout=self.settings.timeout_s,
        )
        return np.asarray(data, dtype=float).reshape(self.n_channels, n_samples)

    def read_mean(self, n_samples: int = 100) -> np.ndarray:
        """Average of a burst per channel, shape ``(n_channels,)``."""
        return self.read_samples(n_samples).mean(axis=1)

    # -- internals ----------------------------------------------------------

    def _require_open(self) -> None:
        if self._task is None:
            raise DaqError("Reader is not open; use `with VoltageReader(...)`")
