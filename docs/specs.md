This revised design specification incorporates your decisions to harden the simulation against the "weird" edge cases we identified. This document now serves as a technical blueprint for a stable, high-fidelity robotic arm simulation.

---

### 1. Physics & Dynamics Refinement
*   **1.1. Mass & Inertia (The "Density-First" Approach):** To improve on standard calculations, we will ignore the high-poly visual mesh for mass properties. Instead, we will assign **Material Densities** (e.g., Aluminum or 3D-printed PLA) to the simplified **Convex Hulls**. The physics engine will then compute the Inertia Tensor based on the simplified geometry, ensuring the center of mass (CoM) and momentum perfectly match the collision behavior.
*   **1.2. Solver Selection:** We will explicitly utilize the **TGS (Temporal Gauss-Seidel)** solver. This is more stable than the default PGS for robotics, especially when using position-driven actuators at a 60Hz frequency.
*   **1.3. Articulation Anchoring:** The `ArticulationRoot` will be flagged as `fixed_base = True`. This anchors the robot to the world origin's local transform, preventing "skating" or micro-drifting of the base.
*   **1.4. Internal Resistance:** The configuration file will expose `joint_friction` and `joint_damping`. 
    *   *Default (Metallic):* Friction = 0.05, Damping = 1.0 (to simulate a lubricated metal gearbox).
*   **1.5. CCD (Continuous Collision Detection):** CCD will be enabled specifically for the **End-Effector and the 3D-printed sensor arm link**. This prevents the thin geometry from "tunneling" through obstacles during high-speed movements.

### 2. Actuation & Control Logic
*   **2.1. Drive Mode:** All joints are configured for **Position Drive**. RMPFlow will output target positions, which the internal PD controller will then resolve into torques.
*   **2.2. Gain Calibration:** Every joint will have explicit `stiffness` and `damping` gains defined in the robot config:
    *   *Stiffness (K):* High enough to minimize gravity sag.
    *   *Damping (D):* Tuned to 2√K (critical damping) to prevent overshoot.
*   **2.3. Temporal Synchronization:** The Physics Step and RMPFlow update frequency are locked at **60 Hz**. This 1:1 ratio ensures that control commands are never "stale" and that the controller doesn't try to command a change faster than the physics engine can resolve it.
*   **2.4. Joint Constraints:** Joint limits are set to **Hard**. In USD, this means `soft_limit` is disabled, ensuring the arm stops exactly at the physical stop, even if RMPFlow attempts to push beyond it.

### 3. Geometry & Contact Specifications
*   **3.1. Collision Filtering:** We will implement **Collision Groups**. 
    *   *Group A (Robot):* Contains all arm links. Self-collision is **disabled** for adjacent links (e.g., forearm cannot hit the elbow) but enabled for non-adjacent links.
    *   *Group B (Environment):* Everything else.
*   **3.2. Contact & Rest Offsets:** 
    *   **Contact Offset:** Set to **0.02m**. (Collisions begin being calculated 2cm before impact to prevent "clipping").
    *   **Rest Offset:** Set to **0.00m**. (Objects will sit flush against each other without a visible "air gap").
*   **3.3. Collision Geometry:** All collision shapes will be generated as **Convex Hulls** rather than raw meshes. This provides the best balance between performance and the "snag-free" movement required for the thin 3D-printed arm.

### 4. System Architecture & Logic
*   **4.1. Intentional Perception Lag:** The simulation will implement a **1-frame render lag** for Camera/Lidar. The "Think" phase will process the image from $T-1$ while the "Act" phase applies to $T$. This mimics the real-world latency of sensor processing pipelines.
*   **4.2. State Restoration:** The `World.Reset()` command will trigger a "Hard Home." It will reset all Articulation joint positions, velocities (to 0.0), and RMPFlow internal state buffers to their default `home_config` values.
*   **4.3. Resource Management:** With a joint count <20, we will use direct **USD Attribute Writes** for targets, as the overhead will be negligible compared to the 60Hz physics step.
*   **4.4. World Orientation:** The stage is defined as **Z-Up** (Standard for ROS and most modern robotics frameworks).

### 5. Edge Case & Fail-Safe Implementation
*   **5.1. Teleportation Guard:** Any manual "Set Translation" on the robot will be wrapped in a function that explicitly calls `set_joint_velocities(0)` and `set_joint_efforts(0)`. This "Cold Reset" prevents the physics engine from interpreting a teleport as an infinite-velocity move.
*   **5.2. Origin Re-centering:** To combat floating-point drift, if the simulation runs for >10,000 steps, the system will check the `ArticulationRoot` transform. If it has drifted >0.1mm from its fixed origin, the script will silently re-snap the base to $(0,0,0)$.
*   **5.3. RMPFlow Timeout:** We will implement a **Stall Monitor**. If the target position is not reached within a configurable `timeout_duration` (e.g., 5 seconds) and the velocity remains below a threshold, the RMPFlow goal is cancelled to prevent the motors from "fighting" an obstacle indefinitely.
*   **5.4. Gripper Bypass:** Since the arm is **not gripping**, the end-effector will be treated as a purely kinematic probe. No friction or grasping physics will be calculated for the terminal link, significantly reducing the complexity of the contact manifold.

---

### How this changes the "Vibe" of your simulation:
By setting **Hard Limits**, **Z-Up**, and **60Hz Sync**, you have moved away from a "video game" style simulation toward a **Deterministic Engineering Tool**. The addition of **Intentional 1-frame lag** is particularly sophisticated—it means that if your controller works in this sim, it is much more likely to work on the real hardware where sensor data is never "instant."