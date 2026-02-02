from abc import ABC, abstractmethod
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from std_msgs.msg import Float32

class RobotInterface(ABC):
    def __init__(self, node: Node):
        self.node = node

    @abstractmethod
    def move_joints(self, joint_positions: list):
        pass

    @abstractmethod
    def move_gripper(self, value: float):
        pass

class SimRobotInterface(RobotInterface):
    """ Interface for Isaac Sim Robot """
    def __init__(self, node: Node):
        super().__init__(node)
        self.pub = node.create_publisher(JointState, '/target_joint_states', 10)
        self.names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "Slider_1", "Slider_2"
        ]

    def move_joints(self, joint_positions: list):
        pass 

    def publish_full_state(self, arm_joints, gripper_val):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = list(arm_joints) + [gripper_val, gripper_val]
        self.pub.publish(msg)

    def move_gripper(self, value: float):
        pass

    def get_current_joints(self):
        return None 

class RealRobotInterface(RobotInterface):
    """ Interface for Real UR3e + Robotiq Hand-E """
    def __init__(self, node: Node):
        super().__init__(node)
        self.pub = node.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        self.pub_gripper = None
        if Float32 is not None:
             self.pub_gripper = self.node.create_publisher(Float32, '/dextel/gripper_cmd', 10)
        
        self.last_gripper_val = -1.0 
        self.current_joints = None
        self.sub_joints = self.node.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg):
        try:
            state_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
            current_pos = []
            for name in self.joint_names:
                if name in state_dict:
                    current_pos.append(state_dict[name])
            
            if len(current_pos) == 6:
                self.current_joints = current_pos
        except Exception:
            pass

    def move_joints(self, joint_positions: list, max_vel: float = 1):
        if JointTrajectory is None: 
            self.node.get_logger().error("CRITICAL: trajectory_msgs.JointTrajectory not imported!")
            return

        min_duration = 0.5
        max_diff = 0.0
        final_goals = list(joint_positions)
        
        if self.current_joints is not None:
            for i in range(6):
                curr = self.current_joints[i]
                tgt = final_goals[i]
                
                diff_raw = tgt - curr
                k = round(diff_raw / (2 * np.pi))
                tgt_new = tgt - k * 2 * np.pi
                final_goals[i] = tgt_new
                
                diff = abs(tgt_new - curr)
                if diff > max_diff:
                    max_diff = diff
        
        duration_sec = max(min_duration, max_diff / max_vel)
        
        # [Debug Logging] Check for timing jitter vs vision noise
        now = self.node.get_clock().now().nanoseconds / 1e9
        dt = now - getattr(self, '_last_cmd_time', now)
        self._last_cmd_time = now
        
        # Log every 1 second (approx every 60 frames)
        if not hasattr(self, '_log_counter'): self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 60 == 0:
            self.node.get_logger().info(
                f"[DEBUG] dt={dt*1000:.1f}ms (expected ~16ms) | Dur={duration_sec:.2f}s | MaxDiff={max_diff:.4f} rad"
            )

        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = final_goals
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        
        msg.points = [point]
        self.pub.publish(msg)

    def get_current_joints(self):
        return self.current_joints

    def move_gripper(self, value: float):
        if self.pub_gripper is None: return

        if abs(value - self.last_gripper_val) < 0.1:
            return
            
        msg = Float32()
        msg.data = float(value)
        self.pub_gripper.publish(msg)
        self.last_gripper_val = value
