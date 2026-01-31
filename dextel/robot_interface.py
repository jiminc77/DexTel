from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node

# Conditional Imports to support environments with partial ROS2 installation (e.g. Sim-only)
try:
    from sensor_msgs.msg import JointState
except ImportError:
    JointState = None

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Header
except ImportError:
    JointTrajectory = None
    JointTrajectoryPoint = None
    Header = None

import numpy as np


class RobotInterface(ABC):
    def __init__(self, node: Node):
        self.node = node

    @abstractmethod
    def move_joints(self, joint_positions: list):
        pass

    @abstractmethod
    def move_gripper(self, value: float):
        """
        value: 0.0 (Close) to 1.0 (Open) or similar.
        Our convention in dextel_node: 
        - 0.0 = Closed (Pinched)
        - 1.0 = Open (Released)
        """
        pass

class SimRobotInterface(RobotInterface):
    """
    Publishes to /target_joint_states for Isaac Sim.
    Message: sensor_msgs/JointState
    Size: 8 (6 Arm + 2 Gripper)
    """
    def __init__(self, node: Node):
        super().__init__(node)
        self.pub = node.create_publisher(JointState, '/target_joint_states', 10)
        
        # Fixed Joint Names for Isaac Sim UR3e
        self.names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "Slider_1", "Slider_2"
        ]

    def move_joints(self, joint_positions: list):
        # We assume joint_positions is [6 arm joints]
        # We need to append gripper state locally or handle it in one go.
        # Design decision: RobotInterface expects 'move_joints' to potentially handle everything 
        # for Sim, but 'move_gripper' updates internal state.
        pass 

    def publish_full_state(self, arm_joints, gripper_val):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.names
        # Sim expects 2 fingers
        # In Sim: 0 (Closed? actually depends on drive)
        # Previous code: 0.0 (Pinched), 0.025 (Open)
        msg.position = list(arm_joints) + [gripper_val, gripper_val]
        self.pub.publish(msg)

    def move_gripper(self, value: float):
        pass

    def get_current_joints(self):
        return None # Sim doesn't subscribe to state yet

# Conditional Imports for Gripper Topic
try:
    from std_msgs.msg import Float32
except ImportError:
    Float32 = None

class RealRobotInterface(RobotInterface):
    """
    Publishes to /scaled_joint_trajectory_controller/joint_trajectory for Real UR3e.
    Controls Gripper via Topic /dextel/gripper_cmd (Float32).
    """
    def __init__(self, node: Node):
        super().__init__(node)
        self.pub = node.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # Gripper Publisher (Talks to simple_robotiq_driver)
        self.pub_gripper = None
        if Float32 is not None:
             self.pub_gripper = self.node.create_publisher(Float32, '/dextel/gripper_cmd', 10)
        
        self.last_gripper_val = -1.0 
        
        # [Safety] Subscribe to actual robot state
        self.current_joints = None
        self.sub_joints = self.node.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        # UR driver publishes all joints. We need to extract the 6 we care about.
        # Order in UR driver: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        # But msg.name might be in different order? Usually it matches.
        # For robustness, we map by name if possible, or assume standard order.
        # Assuming standard UR driver order for now.
        try:
            # Map names to positions
            state_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
            current_pos = []
            for name in self.joint_names:
                if name in state_dict:
                    current_pos.append(state_dict[name])
            
            if len(current_pos) == 6:
                self.current_joints = current_pos
        except Exception:
            pass

    def move_joints(self, joint_positions: list):
        if JointTrajectory is None: 
            self.node.get_logger().error("CRITICAL: trajectory_msgs.JointTrajectory not imported! Cannot move robot.")
            return


        # [Safety] Unwrapping & Deviation Check
        # Jitter Fix: Increase duration to overlap frames (80ms vs 33ms loop)
        duration_sec = 0.08 
        final_goals = list(joint_positions)
        
        if self.current_joints is not None:
            max_diff = 0.0
            
            # Unwrap targets to be closest to current
            for i in range(6):
                curr = self.current_joints[i]
                tgt = final_goals[i]
                
                # Unwrap: tgt_new = tgt - 2pi * round((tgt - curr) / 2pi)
                diff_raw = tgt - curr
                
                # Check if wrapping helps
                # Simple logic: closest multiple of 2pi
                # k = round(diff / 2pi)
                k = round(diff_raw / (2 * np.pi))
                tgt_new = tgt - k * 2 * np.pi
                
                final_goals[i] = tgt_new
                
                diff = abs(tgt_new - curr)
                if diff > max_diff: max_diff = diff
            
            # If deviation > 0.1 rad (~6 degrees), move SLOWLY.
            # This handles the initial "Snap to Home" safely.
            if max_diff > 0.2:
                duration_sec = 2.0
                self.node.get_logger().warn(f"[Safety] Large Deviation ({max_diff:.2f} rad). Moving Slowly (2.0s)...")
            elif max_diff > 0.05:
                duration_sec = 0.5 # Medium correction
        
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = final_goals
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        
        msg.points = [point]
        self.node.get_logger().info(f"[RealRobot] Pub Traj: {final_goals[0]:.2f} (Time: {duration_sec}s)...", throttle_duration_sec=1.0)
        self.pub.publish(msg)

    def get_current_joints(self):
        return self.current_joints

    def move_gripper(self, value: float):
        # Value: 0.0 (Closed) -> 1.0 (Open)
        
        if self.pub_gripper is None: return

        # Optimization: Only publish if state changes significantly
        if abs(value - self.last_gripper_val) < 0.1:
            return
            
        msg = Float32()
        msg.data = float(value)
        self.pub_gripper.publish(msg)
        
        self.last_gripper_val = value
