### Prerequisites (From Step 1/Env Setup)
- Ubuntu 24.04
- ROS 2 Jazzy Installed
- RealSense SDK Installed
- Python 3.10+ environment (Conda `isaac` env recommended)

---

## Vision Teleoperation System (RealSense + HaMeR)

**MediaPipe** for fast 2D ROI detection and **HaMeR (Hand Mesh Recovery)** for accurate 3D pose estimation, fused with **RealSense** depth data.

### 1. Virtual Environment Setup & Dependencies Installation

- Environment Setup
    
    ```bash
    sudo apt install python3-venv
    
    python3 -m venv --system-site-packages venv
    
    # Activate the virtual enviromnemt
    source venv/bin/activate
    
    # Install Core Dependencies
    pip3 install torch torchvision mediapipe==0.10.14 pyrealsense2 opencv-python numpy==1.26.4
    ```
    
- Install HaMeR
    
    ```bash
    cd ~/ros2_ws/src/dextel
    
    # Clone HaMeR repository
    git clone https://github.com/geopavlakos/hamer.git
    
    # Install HaMeR
    cd hamer
    pip3 install -e .
    pip3 install webdataset hydra-core pyrootutils rich smplx==0.1.28 chumpy
    ```
    
- Download Model Data
    
    ```bash
    cd ~/workspace/ros2_ws/src/dextel/hamer
    
    # 1. Download HaMeR Demo Data (Checkpoints)
    wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
    tar -xvf hamer_demo_data.tar.gz
    # 2. Download MANO Hand Model
    mkdir -p _DATA/data/mano
    wget -O _DATA/data/mano/MANO_RIGHT.pkl https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl
    
    cd ~/workspace/ros2_ws/src/dextel/dextel
    # Create a symbolic link to the data
    ln -s ../hamer/_DATA _DATA
    ```

### 2. Vision Node Implementation (`ur3_realsense_hamer.py`)

The `RobustTracker` class integrates the following components:

1.  **Sensor Input**: RealSense D455 RGB-D stream (Aligned).
2.  **ROI Detection**: MediaPipe Hands detects the hand to create a stable bounding box.
3.  **3D Inference**: The **HaMeR** transformer model predicts dense 3D mesh and joint locations from the cropped image.
4.  **Signal Processing**:
    -   **Depth Fusion**: The wrist's true 3D position is determined by sampling the RealSense depth map at the wrist pixel and deprojecting it.
    -   **Smoothing**: `OneEuroFilter` is applied to both the 3D position and the 3D rotation matrix to minimize jitter.
5.  **Logic**:
    -   **Coordinate Frame**: A robust wrist frame is calculated using the Wrist, Index-MCP, and Pinky-MCP joints.
    -   **Pinch Detection**: A Schmitt Trigger (Hysteresis) monitors the Thumb-Index distance for gripper control (Close < 5cm, Open > 10cm).


## Dex-Retargeting Implementation

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
