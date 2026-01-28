# Week 1: Vision & Retargeting Implementation (UR3 Optimized)

## Workflow Overview

This week focuses on building the "Brain" of the teleoperation system: the Vision Pipeline and the Retargeting Engine. By the end of this week, you should be able to move your hand and see consistent, smooth joint angle values ($q_1 \dots q_6$) being published to ROS 2.

### Prerequisites (From Step 1/Env Setup)
- Ubuntu 24.04
- ROS 2 Jazzy Installed
- RealSense SDK Installed
- Python 3.10+ environment (Conda `isaac` env recommended)

---

## Day 3: MediaPipe & Signal Processing Setup

### 1. Dependencies Installation

We will use `mediapipe` for vision and `dex-retargeting` for the optimization logic.

```bash
conda activate isaac

# Install MediaPipe and core math libs
pip install mediapipe opencv-python numpy scipy

# Install Optimization Solver (NLopt)
sudo apt install libnlopt-dev
pip install nlopt

# Install Dex-Retargeting (Clone to modify/inspect if needed, or use as reference)
cd ~/workspace/H1_Project/ai_libs
git clone https://github.com/dexsuite/dex-retargeting.git
pip install -e dex-retargeting
```

### 2. Vision Node Implementation (`ur3_vision.py`)

Create a basic script to extract robust hand vectors.

**Key Goals**:
- Extract 21 landmarks.
- Compute **Wrist Position**.
- Compute **Palm Normal Vector** (Cross product of Index-MCP and Pinky-MCP vectors).
- Compute **Approach Vector** (Wrist to Index-MCP).

```python
# Pseudo-code logic for ur3_vision.py
import mediapipe as mp
import numpy as np

# 1. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. Vector Extraction Logic
def get_hand_vectors(landmarks):
    # Convert landmarks to numpy array (21, 3)
    kps = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    wrist = kps[0]
    index_mcp = kps[5]
    pinky_mcp = kps[17]
    
    # Vector 1: Approach (Wrist -> Index Knuckle) - Main pointing direction
    v_approach = (index_mcp - wrist) / np.linalg.norm(index_mcp - wrist)
    
    # Vector 2: Palm Plane (Index Knuckle -> Pinky Knuckle)
    v_palm_span = (pinky_mcp - index_mcp)
    
    # Vector 3: Normal (Cross Product) - Orientation of the palm
    v_normal = np.cross(v_approach, v_palm_span)
    v_normal /= np.linalg.norm(v_normal)
    
    return wrist, v_approach, v_normal

# 3. Pinch Detection Logic
def get_pinch_distance(landmarks):
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    
    return np.linalg.norm(thumb_tip - index_tip)
```

### 3. Jitter Reduction (Low Pass Filter)

Implement `OneEuroFilter` to stabilize the raw input. This is critical for preventing the robot from shaking.

```python
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0):
        # min_cutoff: Min frequency (lower = smoother, more lag)
        # beta: Speed coefficient (higher = less lag during fast movement)
        self.min_cutoff = min_cutoff
        self.beta = beta
        # ... Implementation ...
```

---

## Day 4: Dex-Retargeting Implementation (The Core)

This is the most critical step. We will adapt the `VectorOptimizer` from `dex-retargeting` to solve for UR3 joint angles.

### 1. URDF Loading

The optimizer needs to know the robot's kinematic structure.

```python
# Load UR3e URDF using a library like 'pinocchio' or 'urdfpy'
# Alternatively, dex-retargeting has its own loader.
import pinocchio as pin

urdf_path = "ur3e.urdf" # Ensure you have the UR3e URDF file
robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
```

### 2. Defining the Target

We map human hand vectors to robot vectors.

- **Human**: Wrist $\rightarrow$ Index MCP
- **Robot**: End-Effector (`tool0`) Z-axis (Approach direction)

- **Human**: Palm Normal
- **Robot**: End-Effector (`tool0`) Y-axis (or X, depending on gripper mounting)

### 3. Implementation of `Optimizer`

```python
from dex_retargeting.optimizer import VectorOptimizer

# Define which link corresponds to the wrist/gripper
target_link_name = "tool0"

# Initialize Optimizer
optimizer = VectorOptimizer(
    robot=robot,
    target_link_names=[target_link_name, target_link_name],
    target_joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
)

# In the loop:
# retargeting_result = optimizer.retarget(human_vectors)
# robot_q = retargeting_result.q
```

---

## Day 5: ROS 2 Integration

Now that we have the math working, we need to publish it to ROS 2 for the simulation to read.

### 1. Node Structure

Create a ROS 2 package `ur3_teleop`.

```bash
cd ~/workspace/ros2_ws/src
ros2 pkg create --build-type ament_python ur3_teleop
```

### 2. Publisher Implementation

- **Topic**: `/target_joint_states`
- **Type**: `sensor_msgs/msg/JointState`
- **Topic**: `/gripper_cmd` (Custom or float)
- **Type**: `std_msgs/msg/Float32` (0.0 = Open, 1.0 = Closed)

```python
# ur3_teleop_node.py snippet

msg = JointState()
msg.name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
msg.position = computed_q_solution # From Optimizer
msg.header.stamp = self.get_clock().now().to_msg()

self.publisher_.publish(msg)
```

---

## Day 6 & 7: Testing & Tuning

### 1. Visual Verification (Rviz2)
Visualize the robot model in Rviz2 and subscribe to `/target_joint_states`.
- **Check 1 (Smoothness)**: Hold your hand still. The robot model in Rviz should NOT vibrate. If it does, decrease `min_cutoff` in OneEuroFilter.
- **Check 2 (Latency)**: Move your hand quickly. The robot should follow without noticeable lag. If laggy, increase `beta` in OneEuroFilter.
- **Check 3 (Singularity)**: Rotate your wrist 180 degrees. The UR3 should utilize its wrist joints to follow, without the whole arm flipping wildly.

### 2. Gripper Tuning
Calibrate the Pinch distance thresholds.
- Measure the distance when your fingers are comfortably touching (~10mm).
- Measure the distance when your hand is relaxed open (~100mm).
- Set thresholds: `CLOSE_THRESH = 15mm`, `OPEN_THRESH = 80mm`.

---

## Deliverables for Week 1
1.  **`ur3_teleop` ROS 2 Package**: Fully functional Python node.
2.  **`config/ur3_retargeting.yml`**: Configuration file for the optimizer/URDF.
3.  **Rviz2 Demo**: A video showing the UR3 ghost model following your hand in real-time.
