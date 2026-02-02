# Phase 3: Real Robot Integration

**Goal**: Deploy the system to the physical UR3e + Robotiq Hand-E safely and effectively.

## 1. Safety First

Moving from simulation to reality requires strict safety measures.
-   **Velocity Scaling**: We limit the `trajectory_msgs` time duration to ensure the robot doesn't move faster than 50% speed during teleop.
-   **Base Limits**: The `retargeting.py` wrapper clamps the base joint to `[-1.6, 1.6]` to prevent the robot from spinning 360 degrees and tearing cables.

---

## 2. Hardware Interfaces

### 2.1. Robot Arm (`RealRobotInterface`)
We communicate with the UR ROS 2 Driver.
-   **Topic**: `/scaled_joint_trajectory_controller/joint_trajectory`
-   **Method**: We bundle the "Target Joints" into a trajectory point with a calculated `time_from_start` based on the distance to travel. This ensures smooth interpolation by the controller.

### 2.2. Gripper (`simple_robotiq_driver.py`)
The standard driver was too slow/complex, so we wrote a direct socket driver.
-   **Protocol**: Connects to Port `63352` on the robot controller.
-   **Commands**: Sends ASCII strings like `SET POS 255` (Close) or `SET POS 0` (Open).
-   **Latency**: <10ms command latency.

---

## 3. Network Setup

1.  **Ethernet**: Connect PC to UR Controller Control Box.
2.  **IP**: Robot IP `137.49.35.26`.
3.  **Connection Test**:
    ```bash
    python3 -m dextel.robot_connection_test
    ```
    Should print "Connection successful!".

---

## 4. Execution Logic

To run on the real robot, we pass the `use_real:=True` flag.

```bash
# Terminal 1: Gripper Driver
python3 -m dextel.simple_robotiq_driver

# Terminal 2: Main Teleop Node
python3 -m dextel.dextel_node --ros-args -p use_real:=True
```

### Operational Check
1.  **Start**: Robot stays in "Wait" mode.
2.  **Homing**: Robot slowly moves to the "Candlestick" (All vertical) home pose.
3.  **Calibration**: User shows hand, presses 'R'.
4.  **Active**: Robot follows hand.
5.  **Safety**: If hand is lost for 3 seconds, Robot performs a "Soft Stop" or re-homes.

---

## 5. Final Capabilities
-   **Latency**: ~50-80ms (End-to-End).
-   **Precision**: Sufficient for block stacking and passing objects.
-   **Robustness**: Hand tracking recovers instantly from occlusion; Robot does not drift.
