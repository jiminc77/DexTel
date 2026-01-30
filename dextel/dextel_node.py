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
        
        self.get_logger().info("Initializing Vision Tracker...")
        self.tracker = RobustTracker()

        self.get_logger().info(f"Initializing Retargeting (URDF: {urdf_path})...")
        try:
            self.retargeting = RetargetingWrapper(urdf_path)
            self.retargeting_enabled = True
        except Exception as e:
            self.get_logger().error(f"Retargeting Init Failed: {e}")
            self.retargeting_enabled = False
            
        self.q_filtered = None
        self.alpha = 0.4 # Smoothing factor (0.0 = infinite smooth/lag, 1.0 = no smooth)

        self.timer = self.create_timer(1.0/30.0, self.control_loop)
        self.frame_count = 0
        self.get_logger().info("DexTel Node Ready.")

        # --- Relative Mapping Components ---
        self.origin_hand_pos = None  # Position of hand when 'R' was pressed
        
        # User-defined Home Configuration (Joint Space)
        # base, shoulder_lift, elbow, w1, w2, w3
        self.home_joints = np.deg2rad([0, -90, -90, -90, 90, 0])
        self.robot_home_pos = None
        self.robot_home_rot = None
        
        self.relative_mode_active = False
        self.movement_scale = 1.5 # Slightly amplified movement for ease

    def control_loop(self):
        # Initialize Home Pose via FK if not set (requires retargeting to be ready)
        if self.robot_home_pos is None and self.retargeting_enabled:
            pos, rot = self.retargeting.compute_fk(self.home_joints)
            self.robot_home_pos = pos
            self.robot_home_rot = rot
            self.get_logger().info(f"Home Pose Computed: {pos}")

        img, state = self.tracker.process_frame()
        
        if img is None:
            self.get_logger().warn("No image from tracker.")
            return

        # --- Handle User Input (for Reset) ---
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            rclpy.shutdown()
            return
        elif key & 0xFF == ord('r'):
            self.relative_mode_active = True
            self.q_filtered = None  # Force Snap / Reset Filter
            
            if state is not None:
                self.origin_hand_pos = state.position
                self.get_logger().info("RESET: Origin Set. Snapping to Home.")
            else:
                self.origin_hand_pos = None
                self.get_logger().info("RESET: No Hand. Going to Home & Waiting...")
        
        # --- Control Logic ---
        publish_dof = None # Joints to publish (if any)
        gripper_val = 0.0  # Default Open

        if self.retargeting_enabled:
            
            if state:
                # [Case 1] Hand Detected
                
                # If we were waiting for a hand (Origin is None), latch it now
                if self.relative_mode_active and self.origin_hand_pos is None:
                     self.origin_hand_pos = state.position
                     self.q_filtered = None # Ensure clean start
                     self.get_logger().info("Hand Detected! Origin Latched.")
                
                # --- Position Mapping ---
                target_pos_rob = None
                
                if self.relative_mode_active and self.origin_hand_pos is not None:
                    # Relative: Home + (Hand - Origin)
                    diff = state.position - self.origin_hand_pos
                    target_pos_rob = self.robot_home_pos + (diff * self.movement_scale)
                else:
                    # Absolute
                    target_pos_rob = state.position
                
                # --- Orientation Mapping ---
                target_rot_rob = state.orientation
                
                # --- Solve IK ---
                q_raw = self.retargeting.solve(target_pos_rob, target_rot_rob)
                if np.isnan(q_raw).any(): q_raw = np.zeros(6)
                
                # --- Smoothing ---
                if self.q_filtered is None:
                    self.q_filtered = q_raw
                else:
                    self.q_filtered = self.alpha * q_raw + (1.0 - self.alpha) * self.q_filtered
                
                publish_dof = self.q_filtered
                
                # Gripper
                gripper_val = 0.8 if state.is_pinched else 0.0
                
            elif self.relative_mode_active and self.origin_hand_pos is None:
                # [Case 2] No Hand, but Reset Active (Waiting) -> Hold Home
                # We use the predetermined home joints directly to ensure exact pose
                publish_dof = self.home_joints
                self.q_filtered = publish_dof # Keep filter synced
                gripper_val = 0.0 # Force Open while waiting

        # --- Publish if we have a target ---
        if publish_dof is not None:
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                "Slider_1", "Slider_2"
            ]
            # Arm (6) + Gripper (2)
            # Ensure publish_dof is list/array of 6
            joint_msg.position = list(publish_dof[:6]) + [gripper_val, gripper_val]
            joint_msg.velocity = [0.0] * 8
            joint_msg.effort = [0.0] * 8
            
            self.pub_joints.publish(joint_msg)
            
        if state:
            is_relative = self.relative_mode_active
            draw_ui_overlay(img, state, 0.0, is_relative)
            
        cv2.imshow("DexTel Control", img)


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
