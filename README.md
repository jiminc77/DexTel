# DexTel: Real-Time UR3e Teleoperation System

## Introduction
DexTel is a real-time teleoperation system for the **UR3e** robot arm using a **Single Realsense D455 Camera** and **AI-based hand tracking**.

The system utilizes **[MediaPipe](https://github.com/google/mediapipe)** and **[HaMeR](https://github.com/geopavlakos/hamer)** for high-fidelity hand tracking and **[Dex-Retargeting](https://github.com/dexsuite/dex-retargeting)** for optimization-based motion mapping, allowing users to control the robot arm intuitively using their own hand movements—**without the need for expensive motion capture gloves, VR controllers, or wearable markers.**

## Key Highlights

*   **Accessible Teleoperation**: Enables high-quality teleoperation using a single RGB-D camera, making the system accessible without the need for high-cost motion capture equipment.
*   **Metric-Scale AI Fusion**: Solves the *scale ambiguity problem* common in monocular 3D vision. The system uniquely fuses **HaMeR's** high-fidelity mesh shape with **RealSense's** precise metric depth, validating the AI's output in the real physical world (mm-level accuracy).
*   **Efficient Two-Stage Inference**: Uses lightweight **MediaPipe** to detect and crop the hand, allowing the heavier **HaMeR** model to focus only on the relevant area for real-time performance.

Key technologies include:
*   **ROS 2 Jazzy**
*   **[MediaPipe](https://github.com/google/mediapipe) + [HaMeR](https://github.com/geopavlakos/hamer)** (Vision & Tracking)
*   **[Dex-Retargeting](https://github.com/dexsuite/dex-retargeting)** (Kinematic Optimization)
*   **Isaac Sim** (Simulation Environment)

## Demos

Here are demonstrations of the system in action:

*   **Simulation Verification**: Validation of control logic in Isaac Sim.

    https://github.com/user-attachments/assets/7a871f9e-7ba4-4467-bb5f-63d9f167c136

*   **Final System**: The complete system controlling the real UR3e robot.
    
    https://github.com/user-attachments/assets/8572fc36-e0e8-4f84-bea0-731d6948e8bb

## Documentation

For detailed information on implementation, installation, and execution, please refer to the **[Documentation](Documentation)** directory.

*   [Project Overview](Documentation/Project_Overview.md)
*   [Technical Manual](Documentation/Technical_Manual.md)
*   [Step 1: Vision & Tracking](Documentation/Step1.md)
*   [Step 2: Retargeting & Simulation](Documentation/Step2.md)
*   [Step 3: Real Robot Integration](Documentation/Step3.md)
