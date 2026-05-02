"""Stub out omni.* and pxr.* so unit tests run without Isaac Sim installed."""
import sys
import types
from unittest.mock import MagicMock


class _AutoMock(types.ModuleType):
    """Module that returns MagicMock for any attribute access."""

    def __getattr__(self, name: str):
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


def _stub(dotted: str) -> None:
    if dotted not in sys.modules:
        mod = _AutoMock(dotted)
        sys.modules[dotted] = mod
        # Wire into parent
        if "." in dotted:
            parent_name, _, child_name = dotted.rpartition(".")
            _stub(parent_name)
            setattr(sys.modules[parent_name], child_name, mod)


for module in (
    "omni",
    "omni.isaac",
    "omni.isaac.core",
    "omni.isaac.core.utils",
    "omni.isaac.core.utils.stage",
    "omni.isaac.core.articulations",
    "omni.isaac.core.objects",
    "omni.isaac.motion_generation",
    "pxr",
    "pxr.Gf",
    "pxr.UsdPhysics",
    "pxr.PhysxSchema",
    "pxr.Usd",
    "pxr.UsdGeom",
):
    _stub(module)
