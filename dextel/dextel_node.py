import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from builtin_interfaces.msg import Time

import numpy as np
import cv2
import time
import os

from dextel.ur3_realsense_hamer import RobustTracker, HandState, draw_ui_overlay
from dextel.retargeting import RetargetingWrapper

class DexTelNode(Node):
    def __init__(self):
        super().__init__('dextel_node')
        
        self.declare_parameter('urdf_path', 'assets/ur3e_hande.urdf')
        param_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(param_path):
            urdf_path = os.path.join(pkg_dir, param_path)
        else:
            urdf_path = param_path
        
        self.pub_joints = self.create_publisher(JointState, '/target_joint_states', 10)
        self.pub_gripper = self.create_publisher(Float64, '/gripper_command', 10)
        
        self.get_logger().info("Initializing Vision Tracker...")
        self.tracker = RobustTracker()

        self.get_logger().info(f"Initializing Retargeting (URDF: {urdf_path})...")
        try:
            self.retargeting = RetargetingWrapper(urdf_path)
            self.retargeting_enabled = True
        except Exception as e:
            self.get_logger().error(f"Retargeting Init Failed: {e}")
            self.retargeting_enabled = False
            
        self.timer = self.create_timer(1.0/30.0, self.control_loop)
        self.get_logger().info("DexTel Node Ready.")

    def control_loop(self):
        img, state = self.tracker.process_frame()
        
        if img is None:
            self.get_logger().warn("No image from tracker.")
            return

        fps = 0.0 
        
        if state and self.retargeting_enabled:
            q_sol = self.retargeting.solve(state.position, state.orientation)
            
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
            joint_msg.position = q_sol.tolist()
            self.pub_joints.publish(joint_msg)
            
            grip_val = 0.8 if state.is_pinched else 0.0
            
            grip_msg = Float64()
            grip_msg.data = grip_val
            self.pub_gripper.publish(grip_msg)
            
        if state:
            draw_ui_overlay(img, state, 0.0)
            
        cv2.imshow("DexTel Control", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DexTelNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tracker.pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
