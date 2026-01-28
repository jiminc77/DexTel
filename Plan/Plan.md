# Real-Time UR3 Teleoperation System - Implementation Plan (ROS 2 Native)

## Project Overview

**Goal**: Implement a real-time (<50ms) teleoperation system where a **UR3e collaborative robot** in Isaac Sim mimics the user's hand movements captured by a **RealSense D455** camera.

**Key Pivot**: Shift from Humanoid (H1) to **6-DOF Manipulator (UR3e)**. This focuses on high-fidelity "Fine Manipulation" tasks (e.g., Pick & Place, Assembly) by leveraging robust optimization-based retargeting.

**Target Environment**:
- **OS**: Ubuntu 24.04 (Noble Numbat)
- **Robot**: **Universal Robots UR3e + Robotiq 2F-85 Gripper**
- **Vision**: Single RealSense D455 (RGB-D)
- **Algorithm**: **MediaPipe Hands** (Vision) + **Dex-Retargeting** (Control)
- **Middleware**: **ROS 2 Jazzy Jalisco** (Native Shared Memory)

---

## System Architecture (UR3 Optimized)

The system utilizes an **Optimization-Based Retargeting** approach to resolve the discrepancy between the human hand (freestyle) and the robot arm (kinematic constraints).

```mermaid
graph TD
    User[User Hand] -->|RGB Stream| MediaPipe[MediaPipe Hands]
    
    subgraph "Vision & Processing Node (60Hz)"
        MediaPipe -->|21 3D Landmarks| LPFilter[Low Pass Filter (OneEuro)]
        LPFilter -->|Smoothed Keypoints| Retargeting
        
        subgraph "Dex-Retargeting (Vector Optimization)"
            target[Vector Calculation]
            target -->|Objective Function| Optimizer[NLopt Optimizer]
            optimizer_logic[Minimize: Direction Error + Similarity] --> Optimizer
            Optimizer -->|Joint Solutions (q1..q6)| Smoother[Temporal Smoothing]
        end
        
        LPFilter -->|Thumb-Index Distance| GripperLogic[Gripper Controller]
    end
    
    Smoother -->|/target_joint_states| ROS2[ROS 2 Shared Memory]
    GripperLogic -->|/gripper_cmd| ROS2
    
    ROS2 -->|Joint Control| Isaac[Isaac Sim (UR3e)]
```

**Total Estimated Latency: ~35ms** (Camera ~15ms + MediaPipe <5ms + Optimizer <1ms + Sim ~16ms).

---

## Key Technology Stack

### 1. Vision: MediaPipe Hands (Google)
- **Role**: Extract 21 3D hand keypoints from RGB images.
- **Why**: 
  - **Speed**: Extremely fast (<5ms on CPU/GPU), leaving ample resources for simulation.
  - **Robustness**: Proven stability for keypoint detection.
  - **Simplicity**: No complex mesh fitting required; direct landmark output is sufficient for manipulator control.

### 2. Control: Dex-Retargeting (Vector Optimization)
- **Role**: Convert human hand poses into robot joint angles ($q_1 \dots q_6$).
- **Algorithm**: **VectorOptimizer** (from `unidexbot/dex-retargeting`).
- **Mechanism**:
  - Instead of forcing the robot to match the absolute XYZ position of the wrist (which causes singularities and reach issues), we optimize the **direction vectors**.
  - **Vector A**: Wrist $\rightarrow$ Index Finger Knuckle (MCP). Matches the robot's end-effector approach vector.
  - **Vector B**: Palm Normal (Index MCP $\times$ Pinky MCP). Matches the gripper's orientation.
- **Advantages**:
  - **Singularity Avoidance**: The optimizer finds the best joint configuration within limits.
  - **Decoupled Grasping**: Hand rotation/translation is mathematically separated from finger grasping.
  - **Stability**: Prevents the "Gimbal Lock" issues common in simple Euler angle mapping.

### 3. Middleware: ROS 2 Jazzy (Shared Memory)
- **Role**: High-speed communication between the Vision Node and Isaac Sim.
- **Configuration**: FastDDS with Shared Memory (SHM) enabled for zero-copy transport.

---

## Implementation Features

### 1. Robust Grasping Logic (Pinch Detection)
- **Logic**: Use the Euclidean distance between **Thumb Tip** and **Index Finger Tip**.
- **Mapping**:
  - Distance < 10mm $\rightarrow$ Gripper CLOSED (100%).
  - Distance > 80mm $\rightarrow$ Gripper OPEN (0%).
  - Linear/Non-linear mapping in between for analog control.
- **Benefit**: This allows the user to rotate their wrist freely while maintaining a firm grasp, as the thumb-index relative distance remains constant during wrist rotation.

### 2. Jitter Reduction (Filtering)
- **Problem**: MediaPipe landmarks inherently jitter (high-frequency noise).
- **Solution**: **OneEuroFilter** (Adaptive Low-Pass Filter).
  - High speed movement $\rightarrow$ Low filtering (High responsiveness).
  - Static/Slow movement $\rightarrow$ High filtering (High stability/No shaking).

---

## Development Roadmap (4 Weeks)

### Week 1: Core Vision & Control Logic
- **Goal**: Control the UR3 robot using hand gestures without simulation visualization.
- **Tasks**:
  1.  Install MediaPipe and `dex-retargeting` libraries.
  2.  Implement `VisionNode` to publish hand vectors.
  3.  Implement `VectorOptimizer` to solve UR3 Inverse Kinematics (IK) in real-time.
  4.  Verify smooth joint output using `rqt_plot`.

### Week 2: Integration & Simulation
- **Goal**: Full loop teleoperation in Isaac Sim.
- **Tasks**:
  1.  Set up UR3e + Robotiq 2F-85 in Isaac Sim (using Isaac Lab assets).
  2.  Connect ROS 2 Bridge (JointState subscribers).
  3.  Tune OneEuroFilter parameters for optimal "Feel" (Latency vs. Smoothness).

### Week 3: Refinement & Demo Construction
- **Goal**: Perform complex manipulation tasks.
- **Tasks**:
  1.  Implement "Relative Control Mode" (freeze robot when hand leaves camera).
  2.  Create demo scenes: **Block Stacking**, **Peg-in-Hole**, **Precision Insertion**.
  3.  Record high-quality demos.

### Week 4: Documentation & Polish
- **Goal**: Finalize project artifacts.
- **Tasks**:
  1.  Code cleanup and commenting.
  2.  Write final `README.md` and `Walkthrough.md`.

---

## Success Metrics
- **Latency**: User feels "connected" to the robot (perceptible delay < 100ms).
- **Stability**: No visible jitter in the robot arm when the user's hand is still.
- **Dexterity**: Successfully perform a "Pick and Place" operation on a 5cm cube.
- **Singularity**: Robot does not "snap" or make sudden erratic movements during full wrist rotation.
