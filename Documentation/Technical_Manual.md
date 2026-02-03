# DexTel Codebase: Logic Deep Dive & Operation Manual
---

## Technical Principles

### 1. System Orchestration: `dextel_node.py`

This module is the "Brain" of the operation. 
It bridges the asynchronous Computer Vision with the synchronous Robot Control.

#### 1.1. Concurrency Model
The system uses a **Producer-Consumer** pattern via `threading`:
-   **Vision Thread (`RobustTracker`)**:
    -   *Role*: Producer.
    -   *Rate*: Variable (~30-45 FPS), dependent on GPU inference time.
    -   *Logic*: Captures RealSense frames $\to$ Inference $\to$ `HandState` object.
-   **Control Loop (Timer @ 60Hz)**:
    -   *Role*: Consumer.
    -   *Rate*: Fixed 60Hz
    -   *Logic*: Fetches the *latest available* `HandState` from the lock-protected shared memory.
    -   *Interpolation*: If a new frame isn't available yet, the **OneEuroFilter** smooths the transition, effectively upsampling the ~30Hz vision signal to the 60Hz robot command signal without step artifacts.

#### 1.2. The Finite State Machine
1.  **HOMING**: Robot moves to a safe start pose.
2.  **WAITING**: Idle state. Checks for hand presence.
3.  **CALIBRATING**:
    -   *Purpose*: To define the coordinate transformation $T_{offset}$.
    -   Records user hand pose $P_{hand\_zero}$ and robot home pose $P_{robot\_zero}$.
    -   During operation: $P_{target} = P_{robot\_zero} + Scale \times (P_{current\_hand} - P_{hand\_zero})$.
    -   Allows operating from *any* comfortable position.
4.  **ACTIVE**: Real-time IK solving and execution.

### 2. Vision Logic: `ur3_realsense_hamer.py`

#### 2.1. Why MediaPipe + HaMeR?
-   **MediaPipe**: Fast 2D ROI detection (hand tracking).
-   **HaMeR**: Accurate 3D Mesh recovery.
-   **Pipeline**: MediaPipe finds ROI $\to$ HaMeR infers Mesh. Optimized for speed.

#### 2.2. Metric Depth Fusion
HaMeR outputs normalized relative space. To control a robot, we need Meters.
1.  Project HaMeR "Wrist" keypoint to 2D pixel $(u, v)$.
2.  Sample **RealSense Depth Map** $D(u, v)$ at that pixel.
3.  **Deprojection (Pinhole Model)**:
    $$ X = (u - c_x) \times Z / f_x $$
    $$ Y = (v - c_y) \times Z / f_y $$
    $$ Z = D(u, v) $$

#### 2.3. Coordinate Frame Construction
We construct a **Gram-Schmidt Orthogonal Frame** on the hand:
-   $\vec{v}_{approach}$ (Red): IndexMCP - Wrist.
-   $\vec{v}_{normal}$ (Green): Palm Normal.
-   $\vec{v}_{side}$ (Blue): Cross Product.

### 3. Retargeting Logic: `retargeting.py`

Solves the "Correspondence Problem" between Human and Robot.

#### 3.1. Vector Optimization
Standard IK matches Position+Quaternion, which fails near singularities.
**`dex-retargeting`** minimizes a Vector Cost Function:
$$ \text{Cost} = w_1 \|\vec{v}_{robot\_z} - \vec{v}_{hand\_side}\|^2 + w_2 \|\vec{v}_{robot\_y} - \vec{v}_{hand\_normal}\|^2 + w_3 \|P_{robot} - P_{target}\|^2 $$
-   **Benefit**: If unreachable, the optimizer "points" in the right direction instead of crashing.

#### 3.2. Safety Constraints
-   **Base Limit**: Clamped to $[-1.6, 1.6]$ rad to prevent cable winding.

### 4. Hardware Interfaces: `robot_interface.py` & `simple_robotiq_driver.py`

-   **RealRobotInterface**: Uses `trajectory_msgs` with dynamic `time_from_start` calculation to ensure the UR controller interpolates smoothly.
-   **SimpleRobotiqDriver**: Uses direct TCP Sockets (Port 63352) for low-latency (<10ms) gripper commands, bypassing ROS driver overhead.

---

## Operation Manual

### 1. Simulation Setup (Isaac Sim)

Use this mode to test logic safely.

**Step 1: Launch Simulation Environment**
This script loads the `ur3e_hande` asset and sets up the ROS 2 Bridge automatically.
```bash
python3 -m dextel.sim_launch
```

**Step 2: Start Control Node**
In a new terminal:
```bash
python3 -m dextel.dextel_node
```

**Step 3: Usage**
1.  **Calibration**: Hold your hand up to the camera. Press **'R'** on the keyboard.
2.  **Wait**: Calibration takes 2 seconds (Yellow status).
3.  **Drive**: Move your hand. The simulated robot should follow.

### 2. Real Robot Deployment

**Prerequisites**:
-   Robot IP: `137.49.35.26`
-   PC connected via Ethernet to Robot Control Box.

**Step 1: Verify Connection**
```bash
python3 -m dextel.robot_connection_test
# Expect: "Connection successful!"
```

**Step 2: Launch UR ROS2 Driver**
This bridges the High-Level ROS 2 topics to the Low-Level Robot Controller.
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=137.49.35.26 launch_rviz:=false
```

**Step 3: Launch Gripper Driver**
Runs the low-latency socket driver.
```bash
python3 -m dextel.simple_robotiq_driver
```

**Step 4: Launch Main Control Node**
The `use_real:=True` flag switches the interface to `RealRobotInterface`.
```bash
python3 -m dextel.dextel_node --ros-args -p use_real:=True
```

**Step 5: Operational Loop**
1.  **Homing**: The robot will slowly move to the "Candlestick" vertical pose.
2.  **Calibrate**: Press **'R'** to zero the hand position.
3.  **Active**: Start teleoperation.
4.  **Safety**: If you withdraw your hand, the robot freezes/homes after 3 seconds.