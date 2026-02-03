# Phase 2: Simulation & Retargeting

**Goal**: Verify kinematic mapping and logic in a safe, simulated environment (Isaac Sim) before touching real hardware.

## 1. Retargeting (Inverse Kinematics)

We utilize **Optimization-Based IK** (`dex-retargeting`) instead of analytic IK to handle the UR3e's limited workspace gracefully.

### Vector Optimization Strategy (`retargeting.py`)
Instead of matching absolute position (which causes singularities), we align **Vectors**:
1.  **Approach Vector**: Aligns Robot Tool Z-Axis with User's Index Finger direction.
2.  **Normal Vector**: Aligns Robot Tool Y-Axis with User's Palm Normal.
3.  **Position**: Aligns Robot Tool Origin with User's Wrist (Relative).

---

## 2. Isaac Sim Setup (`sim_launch.py`)

We created a custom launcher to automate the simulation environment.

### Features
-   **Auto-Load Asset**: Loads `assets/ur3e_hande.usd` automatically.
-   **ROS 2 Bridge**: Creates the Action Graph programmatically to:
    -   Subscribe to `/target_joint_states` (Control).
    -   Publish `/joint_states` (Feedback).
-   **Gripper Config**: Applies high stiffness/damping `DriveAPI` to the gripper joints for stable grasping simulation.

```bash
# Launch Simulation
python3 -m dextel.sim_launch
```

---

## 3. Main Node Logic (`dextel_node.py`)

The `DexTelNode` orchestrates the flow. In simulation mode (`use_real:=False`), it:

1.  Receives `HandState` from Vision.
2.  Calculates `Target Pose` = `RobotHome` + `(HandPos - CalibrationOrigin)`.
3.  Solves IK $\to$ `target_joints`.
4.  Publishes to `/target_joint_states` (read by Isaac Sim Bridge).

### Calibration State
-   On startup, the user presses **'R'**.
-   System records hand pose for 2 seconds.
-   Establishes this as the "Zero Point" to map to the Robot's "Home Pose".
-   Allows comfortable teleoperation regardless of where the user sits.

---

## 4. Verification

1.  **Launch Sim**: `python3 -m dextel.sim_launch`
2.  **Launch Node**: `python3 -m dextel.dextel_node`
3.  **Test**:
    -   Press 'R' to calibrate.
    -   Move hand up/down $\to$ Robot moves up/down.
    -   Rotate wrist $\to$ Robot wrist rotates smoothy.
    -   Pinch $\to$ Sim gripper closes.
