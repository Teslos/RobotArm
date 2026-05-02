"""Quick import smoke test — run before starting the full sim."""
import isaacsim
print("isaacsim: OK")

from isaacsim import SimulationApp
print("SimulationApp: OK")

# Verify omni.isaac.core is reachable (needs SimulationApp running first in real usage)
import omni.isaac.core
print("omni.isaac.core: OK")

import omni.isaac.core.utils.stage
print("omni.isaac.core.utils.stage: OK")

from pxr import UsdPhysics, PhysxSchema, Usd, Gf
print("pxr (USD): OK")

print("\nAll imports OK — Isaac Sim 4.5 ready.")
